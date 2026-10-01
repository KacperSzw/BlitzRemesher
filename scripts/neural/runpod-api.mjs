// Runpod REST v2. Mutating creates are deliberately never retried blindly.
import fs from 'node:fs';
import path from 'node:path';
import { deployment } from './runpod-profile.mjs';
export const GPU = deployment.gpu;
export const IMAGE =
  'runpod/base:1.0.7-cuda1290-ubuntu2404@sha256:c776d549e38023c51a28c25267e029ec2aa53a525c87a934b77694d76572c629';
import { read, write } from './artifacts.mjs';
export { read, write };
export function apiKey(file) {
  const stat = fs.statSync(file);
  if (!stat.isFile() || stat.mode & 0o077)
    throw new Error('API key file must be private (chmod 600)');
  const key = fs.readFileSync(file, 'utf8').trim();
  if (!key || /\s/.test(key)) throw new Error('API key file is empty or malformed');
  return key;
}
// Only callers with repeatable SSH operations use this; never retry API creates.
export async function retrySsh(
  operation,
  wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  attempts = 4,
) {
  for (let attempt = 1; ; ++attempt) {
    try {
      return await operation();
    } catch (error) {
      if ((error.ssh_exit !== 255 && !error.ssh_timeout) || attempt >= attempts) throw error;
      await wait(2000);
    }
  }
}
export class Api {
  constructor(key, fetcher = fetch) {
    this.key = key;
    this.fetcher = fetcher;
  }
  async request(method, resource, body) {
    const response = await this.fetcher('https://api.runpod.io/v2' + resource, {
      method,
      headers: { Authorization: `Bearer ${this.key}`, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });
    if (response.status === 404 && (method === 'GET' || method === 'DELETE')) return null;
    if (!response.ok) {
      const error = new Error(
        `Runpod ${method} ${resource.split('?')[0]}: HTTP ${response.status}`,
      );
      error.status = response.status;
      // Preserve the provider's problem detail locally for failed provisioning.
      // It is not printed with the error or copied into public evidence automatically.
      try {
        const problem = await response.json();
        if (typeof problem.detail === 'string') error.detail = problem.detail;
      } catch {}
      throw error;
    }
    return response.status === 204 ? null : response.json();
  }
  async pods() {
    const pods = [];
    let cursor;
    do {
      const page = await this.request(
        'GET',
        '/pods?limit=100' + (cursor ? '&cursor=' + encodeURIComponent(cursor) : ''),
      );
      pods.push(...page.pods);
      cursor = page.pagination?.nextCursor;
    } while (cursor);
    return pods;
  }
  async quote(dataCenter) {
    const [catalog, centers] = await Promise.all([
      this.request(
        'GET',
        '/catalog/gpus?include=AVAILABILITY&product=POD&count=1&cloud=SECURE&minCudaVersion=' +
          deployment.minimum_cuda,
      ),
      this.request('GET', '/catalog/datacenters?networkVolumeTypes=STANDARD'),
    ]);
    return chooseQuote(catalog.gpus, centers.dataCenters, dataCenter);
  }
}
export function chooseQuote(gpus, centers, dataCenter, profile = deployment) {
  const gpu = gpus.find((g) => g.id === profile.gpu);
  if (
    !gpu?.secure ||
    !(gpu.memory >= profile.catalog_vram_gb) ||
    !(gpu.price?.secure > 0 && gpu.price.secure <= profile.gpu_hourly_usd_cap)
  )
    throw new Error(
      `Secure ${profile.gpu} unavailable within $${profile.gpu_hourly_usd_cap}/GPU-hour cap`,
    );
  const levels = { HIGH: 3, MEDIUM: 2, LOW: 1 };
  const available = (gpu.dataCenters ?? []).filter(
    (d) =>
      (!dataCenter || d.id === dataCenter) &&
      levels[d.availability] &&
      centers.some((c) => c.id === d.id && c.networkVolumeTypes.includes('STANDARD')),
  );
  available.sort(
    (a, b) => levels[b.availability] - levels[a.availability] || a.id.localeCompare(b.id),
  );
  if (!available.length)
    throw new Error(`No Secure ${profile.gpu} location with standard network storage is available`);
  return {
    gpu: gpu.id,
    gpu_hourly_usd: gpu.price.secure,
    data_center: available[0].id,
    quoted_at: new Date().toISOString(),
  };
}
export function podRequest(state, publicKey) {
  const profile = state.deployment ?? deployment;
  return {
    name: state.name,
    cloud: 'SECURE',
    image: IMAGE,
    disk: profile.container_disk_gb,
    gpu: {
      id: profile.gpu,
      count: 1,
      minRamPerGpu: profile.host_ram_gb,
      minVcpuCountPerGpu: profile.vcpus,
      minCudaVersion: profile.minimum_cuda,
    },
    dataCenterIds: [state.quote.data_center],
    mounts: { network: [{ volumeId: state.volume_id, path: '/workspace' }] },
    ports: ['22/tcp'],
    startSsh: true,
    startJupyter: false,
    env: {
      PUBLIC_KEY: publicKey.trim(),
      ...(profile.graphics ? { NVIDIA_DRIVER_CAPABILITIES: 'compute,utility,graphics' } : {}),
    },
  };
}
export function verifyPod(pod, profile = deployment) {
  // Pod gpu.memory is allocated HOST RAM; catalog memory and nvidia-smi verify VRAM.
  if (
    pod.cloud !== 'SECURE' ||
    pod.gpu?.id !== profile.gpu ||
    pod.gpu.count !== 1 ||
    !(pod.gpu.memory >= profile.host_ram_gb) ||
    !(pod.gpu.vcpuCount >= profile.vcpus)
  )
    throw new Error('Allocated Pod does not match approved GPU/CPU/RAM requirements');
  // `cost` includes the container disk. Allow its documented $0.10/GB-month separately.
  if (
    !Number.isFinite(pod.cost) ||
    pod.cost <= 0 ||
    pod.cost > profile.gpu_hourly_usd_cap + (profile.container_disk_gb * 0.1) / (30 * 24) + 0.00001
  )
    throw new Error('Allocated Pod exceeds approved hourly compute/storage rate');
}
export function terminationDue(state, now, cancelled = false) {
  return (
    cancelled ||
    now >= state.deadline_ms ||
    (!state.setup_complete && now >= state.setup_deadline_ms)
  );
}
export function rentalDeadlines(started, deadline, profile = deployment) {
  const maximum =
    started +
    (profile.setup_minutes + profile.training_minutes + profile.collection_minutes) * 60000;
  if (deadline === undefined) deadline = maximum;
  if (
    !Number.isFinite(deadline) ||
    deadline <= started + profile.collection_minutes * 60000 ||
    deadline > maximum
  )
    throw new Error(
      'Rental deadline must leave collection time and stay within the approved setup/training budget',
    );
  const training = deadline - profile.collection_minutes * 60000;
  return {
    started_at: started,
    setup_deadline_ms: Math.min(started + profile.setup_minutes * 60000, training),
    training_minutes: profile.training_minutes,
    training_deadline_ms: training,
    deadline_ms: deadline,
  };
}
export class Rental {
  constructor(api, state, save, now = Date.now) {
    this.api = api;
    this.state = state;
    this.save = save;
    this.now = now;
  }
  commit(extra) {
    Object.assign(this.state, extra);
    this.save(this.state);
  }
  async volume() {
    const s = this.state;
    if (s.volume_id) return;
    if (s.volume_requested) {
      const result = await this.api.request('GET', '/network-volumes');
      const matches = result.networkVolumes.filter((v) => v.name === s.name);
      if (matches.length !== 1)
        throw new Error('Volume create outcome unresolved; refusing another create');
      this.commit({ volume_id: matches[0].id });
      return;
    }
    if (this.now() >= s.setup_deadline_ms) throw new Error('Setup deadline reached');
    this.commit({ volume_requested: true });
    try {
      const v = await this.api.request('POST', '/network-volumes', {
        name: s.name,
        size: (s.deployment ?? deployment).network_volume_gb,
        dataCenter: s.quote.data_center,
        type: 'STANDARD',
      });
      this.commit({ volume_id: v.id });
    } catch (error) {
      if ([400, 401, 403, 404, 409, 422, 429].includes(error.status))
        this.commit({ volume_rejected: true });
      throw error;
    }
  }
  async pod(publicKey) {
    const s = this.state;
    if (s.pod_id) return;
    if (this.now() >= s.setup_deadline_ms) throw new Error('Setup deadline reached');
    if (s.pod_requested) {
      const matches = (await this.api.pods()).filter((p) => p.name === s.name);
      if (matches.length !== 1)
        throw new Error('Pod create outcome unresolved; refusing another create');
      this.commit({ pod_id: matches[0].id });
      return;
    }
    this.commit({ pod_requested: true, pod_requested_at: this.now() });
    try {
      const pod = await this.api.request('POST', '/pods', podRequest(s, publicKey));
      this.commit({ pod_id: pod.id });
    } catch (error) {
      // A documented client rejection has a known outcome. A disconnect or
      // server error may still have created a Pod and must be reconciled.
      if ([400, 401, 403, 404, 409, 422, 429].includes(error.status))
        this.commit({ pod_rejected: true, pod_rejection_status: error.status });
      throw error;
    }
  }
  async terminate() {
    const s = this.state;
    // Reconcile by durable unique name even if creation's response was lost.
    const pods = (await this.api.pods()).filter((p) => p.id === s.pod_id || p.name === s.name);
    if (s.pod_id && !pods.some((p) => p.id === s.pod_id)) {
      const known = await this.api.request('GET', '/pods/' + encodeURIComponent(s.pod_id));
      if (known) pods.push(known);
    }
    if (s.pod_requested && !s.pod_rejected && !s.pod_id && !pods.length)
      throw new Error('Pod create outcome remains unresolved; watchdog must keep reconciling');
    for (const pod of pods) {
      await this.api.request('DELETE', '/pods/' + encodeURIComponent(pod.id));
      const current = await this.api.request('GET', '/pods/' + encodeURIComponent(pod.id));
      if (current && current.status !== 'TERMINATED')
        throw new Error('Pod deletion is not yet confirmed');
    }
    this.commit({ compute_terminated: true, terminated_at: this.now() });
  }
  async cleanupVolume(verified) {
    if (!this.state.compute_terminated)
      throw new Error('Terminate compute before deleting storage');
    const neverAttached =
      !this.state.pod_id && (this.state.pod_rejected || !this.state.pod_requested);
    if (!verified && !neverAttached) return; // The sole recovery copy stays on persistent storage.
    if (!this.state.volume_id && this.state.volume_requested && !this.state.volume_rejected)
      await this.volume();
    if (this.state.volume_id)
      await this.api.request(
        'DELETE',
        '/network-volumes/' + encodeURIComponent(this.state.volume_id),
      );
    this.commit({ volume_deleted: true });
  }
}
