import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const GiB = 1024 ** 3;

function positive(text) {
  if (!/^\d+$/.test(text ?? '')) return null;
  const value = Number(text);
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}

export function cpuSetSize(text) {
  if (!text?.trim()) return null;
  if (!/^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$/.test(text.trim())) return null;
  const ranges = text
    .trim()
    .split(',')
    .map((part) => part.split('-').map(Number));
  if (
    ranges.some((range) => range.length > 2 || range.some((n) => !Number.isSafeInteger(n) || n < 0))
  )
    return null;
  ranges.sort((a, b) => a[0] - b[0]);
  let count = 0,
    last = -1;
  for (const [start, end = start] of ranges) {
    if (end < start || start <= last) return null;
    count += end - start + 1;
    last = end;
  }
  return count || null;
}

// Scheduling estimates, not a hard process-memory guarantee. One heavyweight
// compiler reserves 6 GiB, each additional compiler 2 GiB, with 2 GiB headroom.
export function planBuildResources({ affinity, hostAvailableBytes, groups = [], malformed = [] }) {
  const cpus = [Number.isInteger(affinity) && affinity > 0 ? affinity : 1];
  const available = [
    Number.isFinite(hostAvailableBytes) && hostAvailableBytes > 0 ? hostAvailableBytes : 0,
  ];
  for (const group of groups) {
    if (group.cpu_count !== null && group.cpu_count !== undefined)
      cpus.push(Math.max(1, Math.floor(group.cpu_count)));
    if (group.available_bytes !== null && group.available_bytes !== undefined)
      available.push(Math.max(0, group.available_bytes));
  }
  const cpuLimit = Math.min(...cpus);
  const availableBytes = Math.min(...available);
  const memoryJobs = 1 + Math.floor(Math.max(0, availableBytes - 8 * GiB) / (2 * GiB));
  const jobs = malformed.length ? 1 : Math.max(1, Math.min(cpuLimit, memoryJobs, 8));
  return {
    version: 1,
    jobs,
    cuda_jobs: Math.min(jobs, 2),
    heavy_jobs: 1,
    cpu_limit: cpuLimit,
    available_bytes: availableBytes,
    affinity,
    groups,
    malformed,
    assumptions: {
      max_jobs: 8,
      heavy_bytes: 6 * GiB,
      additional_job_bytes: 2 * GiB,
      headroom_bytes: 2 * GiB,
    },
  };
}

export function detectBuildResources({
  read = (file) => fs.readFileSync(file, 'utf8'),
  affinity = os.availableParallelism?.() ?? os.cpus().length,
} = {}) {
  const malformed = [],
    groups = [];
  const optional = (file) => {
    try {
      return read(file).trim();
    } catch (error) {
      if (!['ENOENT', 'ENOTDIR'].includes(error.code)) malformed.push(file);
      return null;
    }
  };
  const meminfo = optional('/proc/meminfo');
  const hostAvailable = positive(meminfo?.match(/^MemAvailable:\s+(\d+) kB$/m)?.[1]);
  if (hostAvailable === null) malformed.push('/proc/meminfo:MemAvailable');
  const locations = new Map([
    ['/sys/fs/cgroup', 'v2'],
    ['/sys/fs/cgroup/cpu', 'cpu'],
    ['/sys/fs/cgroup/cpu,cpuacct', 'cpu'],
    ['/sys/fs/cgroup/cpuset', 'cpuset'],
    ['/sys/fs/cgroup/memory', 'memory'],
  ]);
  // Container cgroup namespaces expose their limits at the root. For nested
  // processes, also inspect every visible ancestor; a parent can be tighter.
  const cgroup = optional('/proc/self/cgroup');
  if (cgroup === null) malformed.push('/proc/self/cgroup');
  for (const line of cgroup?.split('\n') ?? []) {
    const match = line.match(/^\d+:([^:]*):(\/.*)$/);
    if (!match) {
      malformed.push('/proc/self/cgroup:entry');
      continue;
    }
    const controllers = match[1].split(',');
    if (match[2].split('/').includes('..')) continue; // namespace-relative ancestors are not accessible
    const roots =
      match[1] === ''
        ? [['/sys/fs/cgroup', 'v2']]
        : [...locations].filter(([, kind]) => controllers.includes(kind));
    for (const [root, kind] of roots) {
      for (let dir = path.join(root, match[2]); dir.startsWith(root + '/'); dir = path.dirname(dir))
        locations.set(dir, kind);
    }
  }
  for (const [dir, kind] of locations) {
    const group = { path: dir, cpu_count: null, available_bytes: null };
    if (kind === 'v2' || kind === 'cpu') {
      const quotaFile = dir + (kind === 'v2' ? '/cpu.max' : '/cpu.cfs_quota_us');
      const text = optional(quotaFile);
      if (text !== null) {
        const [quota, period] =
          kind === 'v2' ? text.split(/\s+/) : [text, optional(dir + '/cpu.cfs_period_us')];
        if (quota === 'max' && positive(period) === null) malformed.push(quotaFile);
        else if (quota !== 'max' && quota !== '-1') {
          const q = positive(quota),
            p = positive(period);
          if (q === null || p === null) malformed.push(quotaFile);
          else group.cpu_count = q / p;
        }
      }
    }
    if (kind === 'v2' || kind === 'cpuset') {
      const file = dir + (kind === 'v2' ? '/cpuset.cpus.effective' : '/cpuset.cpus');
      const text = optional(file);
      if (text) {
        const count = cpuSetSize(text);
        if (count === null) malformed.push(file);
        else group.cpu_count = Math.min(group.cpu_count ?? Infinity, count);
      }
    }
    if (kind === 'v2' || kind === 'memory') {
      const file = dir + (kind === 'v2' ? '/memory.max' : '/memory.limit_in_bytes');
      const text = optional(file);
      if (text !== null && text !== 'max') {
        // cgroup v1 uses an enormous sentinel for unlimited memory.
        const unlimited = /^\d+$/.test(text) && BigInt(text) >= BigInt(Number.MAX_SAFE_INTEGER);
        if (!unlimited) {
          const limit = positive(text);
          const used = optional(
            dir + (kind === 'v2' ? '/memory.current' : '/memory.usage_in_bytes'),
          );
          if (limit === null || !/^\d+$/.test(used ?? '') || !Number.isSafeInteger(Number(used)))
            malformed.push(file);
          else group.available_bytes = Math.max(0, limit - Number(used));
        }
      }
    }
    if (group.cpu_count !== null || group.available_bytes !== null) groups.push(group);
  }
  if (groups.length === 0 && cgroup) malformed.push('/sys/fs/cgroup:constraints unavailable');
  return planBuildResources({
    affinity,
    hostAvailableBytes: (hostAvailable ?? 0) * 1024,
    groups,
    malformed,
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) throw new Error('Usage: build-resources.mjs OUTPUT_JSON');
  fs.writeFileSync(process.argv[2], JSON.stringify(detectBuildResources(), null, 2) + '\n');
}
