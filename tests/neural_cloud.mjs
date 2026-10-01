import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  Api,
  Rental,
  IMAGE,
  chooseQuote as quoteFor,
  podRequest,
  verifyPod as verifyFor,
  terminationDue,
  rentalDeadlines as deadlinesFor,
  retrySsh,
  storageMode,
  inputArchive,
} from '../scripts/neural/runpod-api.mjs';
import { profiles, verifyDevice } from '../scripts/neural/runpod-profile.mjs';
import {
  coreValidationAuthorization,
  coreValidationBudget,
  coreValidationProfiles,
  coreValidationSteps,
  validateCoreTestLog,
} from '../scripts/neural/core-validation.mjs';
import { runCoreValidation } from '../scripts/neural/core-validation-job.mjs';
import {
  readinessModel,
  readinessFixtures,
  readinessManifest,
  runTeacherReadiness,
} from '../scripts/neural/teacher-readiness.mjs';
// Teacher commands and their frozen manifest use repository-relative paths.
// CTest starts this isolated Node process in its build directory.
process.chdir(fileURLToPath(new URL('../', import.meta.url)));
const checksum = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
function modelFixture(file, architecture = 4, width = 64) {
  const weights = width * 129 + width * (width + 1) + 12 * (width + 1);
  const bytes = Buffer.alloc(24 + weights * 4);
  bytes.write('BLZNET02');
  bytes.writeUInt32LE(architecture, 8);
  bytes.writeUInt32LE(weights, 12);
  bytes.writeUInt32LE(width, 20);
  fs.writeFileSync(file, Buffer.concat([bytes, Buffer.from(checksum(bytes))]));
}
const deployment = {
    ...profiles.blackwell,
    gpu_hourly_usd_cap: 2.5,
    host_ram_gb: 32,
    vcpus: 16,
    setup_minutes: 30,
    training_minutes: 120,
    collection_minutes: 10,
  },
  GPU = deployment.gpu;
const rentalDeadlines = (start, end) => deadlinesFor(start, end, deployment);
const chooseQuote = (gpus, centers, center) => quoteFor(gpus, centers, center, deployment),
  verifyPod = (pod) => verifyFor(pod, deployment);
const gpu = {
  id: GPU,
  memory: 96,
  secure: true,
  price: { secure: 2.09 },
  dataCenters: [{ id: 'EU-1', availability: 'HIGH' }],
};
const centers = [{ id: 'EU-1', networkVolumeTypes: ['STANDARD'] }];
const initial = () => ({
  name: 'test-unique',
  deployment,
  quote: chooseQuote([gpu], centers),
  setup_deadline_ms: 1800000,
  deadline_ms: 7200000,
});
test('input archives accept compressed and legacy bundles without arbitrary paths', () => {
  assert.equal(inputArchive({}), 'input.tar');
  for (const name of ['input.tar', 'input.tar.gz'])
    assert.equal(inputArchive({ archive_name: name }), name);
  for (const name of ['', '../input.tar', '/input.tar.gz', 'other.tar', 4])
    assert.throws(() => inputArchive({ archive_name: name }), /archive name/);
});
test('rental timings preserve transitions across restart without counting repeated observations', () => {
  let time = 100,
    saved;
  const rental = new Rental(
    null,
    initial(),
    (state) => {
      saved = structuredClone(state);
    },
    () => time,
  );
  rental.commit({ phase: 'provisioning' });
  time = 150;
  rental.commit({ pod_id: 'fixture' });
  rental.commit({ phase: 'provisioning' });
  time = 300;
  rental.commit({ phase: 'ssh' });
  const resumed = new Rental(
    null,
    saved,
    (state) => {
      saved = structuredClone(state);
    },
    () => time,
  );
  time = 400;
  resumed.commit({ phase: 'provisioning' });
  time = 450;
  resumed.commit({ phase: 'upload' });
  assert.deepEqual(saved.phase_history, [
    { phase: 'provisioning', at: 100 },
    { phase: 'ssh', at: 300 },
    { phase: 'provisioning', at: 400 },
    { phase: 'upload', at: 450 },
  ]);
  assert.equal(saved.pod_id, 'fixture');
  assert.equal(saved.deadline_ms, initial().deadline_ms);
});
test('repeatable SSH calls recover transport failures, preserve command failures and stop retrying', async () => {
  let calls = 0,
    waits = 0;
  const network = Object.assign(new Error('connection timeout'), { ssh_exit: 255 });
  const result = await retrySsh(
    async () => {
      if (++calls < 3) throw network;
      return 'ready';
    },
    async () => {
      ++waits;
    },
  );
  assert.equal(result, 'ready');
  assert.equal(calls, 3);
  assert.equal(waits, 2);
  for (const code of [1, 2]) {
    calls = 0;
    const command = Object.assign(new Error('command failed'), { ssh_exit: code });
    await assert.rejects(
      retrySsh(
        async () => {
          ++calls;
          throw command;
        },
        async () => {},
      ),
      /command failed/,
    );
    assert.equal(calls, 1);
  }
  calls = 0;
  await assert.rejects(
    retrySsh(
      async () => {
        ++calls;
        throw network;
      },
      async () => {},
      3,
    ),
    /connection timeout/,
  );
  assert.equal(calls, 3);
});
test('recovery can shorten a rental to its original cutoff without extending the budget', () => {
  const started = 12340000,
    originalEnd = started + 90 * 60000;
  const limits = rentalDeadlines(started, originalEnd);
  assert.equal(limits.deadline_ms, originalEnd);
  assert.equal(limits.training_deadline_ms, originalEnd - 10 * 60000);
  assert.ok(limits.setup_deadline_ms <= started + 30 * 60000);
  for (const minutes of [15, 45, 160]) {
    const d = rentalDeadlines(started, started + minutes * 60000);
    assert.ok(d.setup_deadline_ms <= d.training_deadline_ms);
    assert.ok(d.training_deadline_ms < d.deadline_ms);
  }
  assert.equal(rentalDeadlines(started).training_minutes, 120);
  for (const deadline of [NaN, Infinity, started, started + 10 * 60000, started + 161 * 60000])
    assert.throws(() => rentalDeadlines(started, deadline));
});
test('status succeeds with optional state files absent and reports every available record', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-cloud-status-'));
  const cli = fileURLToPath(new URL('../scripts/neural/runpod.mjs', import.meta.url));
  try {
    fs.writeFileSync(directory + '/rental.json', JSON.stringify({ name: 'fixture-rental' }));
    for (const collected of [false, true]) {
      if (collected)
        fs.writeFileSync(directory + '/collection.json', JSON.stringify({ verified: true }));
      const result = spawnSync(process.execPath, [cli, 'status', directory], { encoding: 'utf8' });
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /fixture-rental/);
      assert.equal(result.stdout.includes('collection.json'), collected);
    }
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
class MockApi {
  constructor() {
    this.calls = [];
    this.volumes = [];
    this.resources = [];
    this.lost = false;
    this.deletionFails = false;
  }
  async pods() {
    return this.resources.map((p) => ({ ...p }));
  }
  async request(method, uri, body) {
    this.calls.push({ method, uri, body });
    if (uri === '/network-volumes') {
      if (method === 'GET') return { networkVolumes: this.volumes };
      const v = { ...body, id: 'volume1' };
      this.volumes.push(v);
      return v;
    }
    if (uri === '/pods' && method === 'POST') {
      const pod = { ...body, id: 'pod1' };
      this.resources.push(pod);
      if (this.lost) throw new Error('connection lost after create');
      return pod;
    }
    if (uri.startsWith('/pods/')) {
      const id = decodeURIComponent(uri.slice(6));
      if (method === 'DELETE') {
        if (this.deletionFails) throw new Error('temporary API failure');
        this.resources = this.resources.filter((p) => p.id !== id);
        return null;
      }
      return this.resources.find((p) => p.id === id) ?? null;
    }
    if (uri.startsWith('/network-volumes/') && method === 'DELETE') {
      this.volumes = [];
      return null;
    }
    throw new Error('Unexpected mock request');
  }
}
test('reject wrong GPUs, unknown prices, insufficient VRAM and unavailable storage', () => {
  for (const invalid of [
    { ...gpu, id: 'NVIDIA GeForce RTX 5090' },
    { ...gpu, id: GPU + ' MIG 2g.48gb' },
    { ...gpu, memory: 48 },
    { ...gpu, secure: false },
    { ...gpu, price: { secure: 2.51 } },
    { ...gpu, price: { secure: NaN } },
    { ...gpu, dataCenters: [{ id: 'EU-1', availability: 'NONE' }] },
  ])
    assert.throws(() => chooseQuote([invalid], centers));
  assert.throws(() => chooseQuote([gpu], []));
  assert.throws(() => chooseQuote([gpu], centers, 'unavailable-location'));
  assert.equal(chooseQuote([gpu], centers, 'EU-1').data_center, 'EU-1');
  for (const rate of [1.9, 2.09, 2.5])
    assert.equal(chooseQuote([{ ...gpu, price: { secure: rate } }], centers).gpu_hourly_usd, rate);
});
test('Pod request is pinned and forwards only the public key', () => {
  const s = { ...initial(), volume_id: 'volume1' },
    body = podRequest(s, 'ssh-ed25519 public-test\n');
  assert.equal(body.image, IMAGE);
  assert.match(body.image, /@sha256:[a-f0-9]{64}$/);
  assert.equal(body.gpu.count, 1);
  assert.equal(body.cloud, 'SECURE');
  assert.deepEqual(body.env, { PUBLIC_KEY: 'ssh-ed25519 public-test' });
  assert.deepEqual(body.ports, ['22/tcp']);
  assert.equal(body.mounts.network[0].volumeId, 'volume1');
});
test('container storage is explicit, core-only and does not require network-volume availability', async () => {
  const state = { ...initial(), experiment: 'core-validation', storage_mode: 'container' };
  assert.equal(storageMode(initial()), 'network');
  assert.equal(storageMode(state), 'container');
  assert.equal(Object.hasOwn(podRequest(state, 'public'), 'mounts'), false);
  assert.equal(podRequest(state, 'public').disk, deployment.container_disk_gb);
  const withoutVolumes = [{ id: 'EU-1', networkVolumeTypes: [] }];
  assert.throws(() => quoteFor([gpu], withoutVolumes, undefined, deployment));
  assert.equal(
    quoteFor([gpu], withoutVolumes, undefined, deployment, 'container').data_center,
    'EU-1',
  );
  const api = new MockApi();
  let durable;
  let rental = new Rental(
    api,
    state,
    (s) => {
      durable = structuredClone(s);
    },
    () => 100,
  );
  await rental.volume();
  assert.equal(api.calls.length, 0);
  await rental.pod('public');
  assert.equal(durable.storage_mode, 'container');
  assert.equal(Object.hasOwn(api.calls[0].body, 'mounts'), false);
  rental = new Rental(
    api,
    durable,
    () => {},
    () => 200,
  );
  await rental.terminate(true);
  await rental.cleanupVolume(true);
  assert.equal(durable.compute_terminated, true);
  assert.equal(durable.results_verified, true);
  assert.equal(durable.ephemeral_storage_lost, false);
  assert.equal(durable.container_storage_preserved, false);
  assert.ok(api.calls.every((call) => !call.uri.startsWith('/network-volumes')));
});
test('malformed storage blocks provisioning while owned Pod termination remains available', async () => {
  for (const changed of [
    { storage_mode: null },
    { storage_mode: 'typo' },
    { storage_mode: 'container', experiment: 'gpu-refactor' },
    { storage_mode: 'container', volume_id: 'volume1' },
    { storage_mode: 'container', volume_requested: true },
  ]) {
    const api = new MockApi(),
      state = {
        ...initial(),
        experiment: 'core-validation',
        storage_mode: 'container',
        ...changed,
      };
    const rental = new Rental(
      api,
      state,
      () => {},
      () => 200,
    );
    assert.throws(() => podRequest(state, 'public'), /storage|network-volume/);
    await assert.rejects(rental.volume(), /storage|network-volume/);
    await assert.rejects(rental.pod('public'), /storage|network-volume/);
    assert.equal(api.calls.length, 0);
    state.pod_id = 'owned';
    state.pod_requested = true;
    api.resources = [
      { id: 'owned', name: state.name },
      { id: 'unrelated', name: 'training' },
    ];
    api.volumes = [{ id: 'volume1', name: state.name }];
    await rental.terminate();
    assert.deepEqual(
      api.resources.map((pod) => pod.id),
      ['unrelated'],
    );
    assert.equal(state.compute_terminated, true);
    await assert.rejects(rental.cleanupVolume(true), /storage|network-volume/);
    assert.equal(api.volumes.length, 1);
  }
});
test('actual allocation must satisfy hardware and rate caps', () => {
  const pod = {
    cloud: 'SECURE',
    cost: 2.1,
    gpu: { id: GPU, count: 1, memory: deployment.host_ram_gb, vcpuCount: deployment.vcpus },
  };
  verifyPod(pod);
  // Runpod's Pod gpu.memory is host RAM; 32 GB is valid with a 96 GB GPU.
  verifyPod({ ...pod, gpu: { ...pod.gpu, memory: 64, vcpuCount: deployment.vcpus * 2 } });
  for (const wrong of [
    { ...pod, cost: 2.52 },
    { ...pod, cost: undefined },
    { ...pod, gpu: { ...pod.gpu, id: GPU + ' MIG 2g.48gb' } },
    { ...pod, gpu: { ...pod.gpu, count: 2 } },
    { ...pod, gpu: { ...pod.gpu, vcpuCount: deployment.vcpus - 1 } },
    { ...pod, gpu: { ...pod.gpu, memory: undefined } },
  ])
    assert.throws(() => verifyPod(wrong));
});
test('remote hardware validation requires a full device and a valid compatible driver', () => {
  const profile = {
    gpu: 'Fixture GPU',
    compute_capability: '12.0',
    device_memory_mib: 90000,
    minimum_driver: [575, 51, 3],
  };
  for (const memory of [95000, 98000])
    for (const driver of ['575.51.03', '575.52.01', '580.0.0'])
      assert.equal(
        verifyDevice(`Fixture GPU, 12.0, ${memory}, ${driver}`, profile).memory_mib,
        memory,
      );
  for (const row of [
    'Fixture GPU, 12.0, 48000, 580.0.0',
    'Fixture GPU MIG, 12.0, 98000, 580.0.0',
    'Fixture GPU, 9.0, 98000, 580.0.0',
    'Fixture GPU, 12.0, N/A, 580.0.0',
    'Fixture GPU, 12.0, 98000, N/A',
    'Fixture GPU, 12.0, 98000, 574.99.99',
    'Fixture GPU, 12.0, 98000, 575.51.02',
    'Fixture GPU, 12.0, 98000, 580.0.0\nFixture GPU, 12.0, 98000, 580.0.0',
  ])
    assert.throws(() => verifyDevice(row, profile));
  assert.equal(deployment.gpu, 'NVIDIA RTX PRO 6000 Blackwell Server Edition');
  assert.equal(
    verifyDevice(`${deployment.gpu}, 12.0, 97280, 580.0.0`, deployment).compute_capability,
    '12.0',
  );
});
test('each selected profile binds quote, Pod allocation, native architecture and rate cap', () => {
  for (const profile of Object.values(profiles)) {
    const entry = {
      ...gpu,
      id: profile.gpu,
      memory: profile.catalog_vram_gb,
      price: { secure: profile.gpu_hourly_usd_cap - 0.05 },
    };
    const quote = quoteFor([entry], centers, undefined, profile),
      state = { ...initial(), deployment: profile, quote };
    const body = podRequest(state, 'public');
    assert.equal(body.gpu.id, profile.gpu);
    assert.equal(quote.gpu, profile.gpu);
    assert.equal(Number(profile.compute_capability) * 10, profile.cuda_architecture);
    assert.throws(() =>
      quoteFor(
        [{ ...entry, price: { secure: profile.gpu_hourly_usd_cap + 0.01 } }],
        centers,
        undefined,
        profile,
      ),
    );
    verifyDevice(
      `${profile.gpu}, ${profile.compute_capability}, ${profile.device_memory_mib + 1000}, 580.0.0`,
      profile,
    );
    verifyFor(
      {
        cloud: 'SECURE',
        cost: profile.gpu_hourly_usd_cap,
        gpu: { id: profile.gpu, count: 1, memory: profile.host_ram_gb, vcpuCount: profile.vcpus },
      },
      profile,
    );
  }
});
test('lost create response is reconciled after restart without a second rental or deadline reset', async () => {
  const api = new MockApi();
  api.lost = true;
  let durable;
  const rental = new Rental(
    api,
    initial(),
    (s) => {
      durable = structuredClone(s);
    },
    () => 100,
  );
  await rental.volume();
  await assert.rejects(rental.pod('public'), /connection lost/);
  assert.equal(durable.pod_requested, true);
  assert.equal(durable.pod_id, undefined);
  const resumed = new Rental(
    api,
    durable,
    (s) => {
      durable = structuredClone(s);
    },
    () => 500,
  );
  await resumed.pod('public');
  assert.equal(durable.pod_id, 'pod1');
  assert.equal(api.calls.filter((c) => c.method === 'POST' && c.uri === '/pods').length, 1);
  assert.equal(durable.deadline_ms, 7200000);
  api.resources.push({ id: 'unrelated', name: 'someone-elses-pod' });
  await resumed.terminate();
  assert.equal(durable.compute_terminated, true);
  assert.deepEqual(
    api.resources.map((p) => p.id),
    ['unrelated'],
  );
});
test('ambiguous provisioning must not be reported as terminated when no response is visible yet', async () => {
  const api = new MockApi(),
    state = { ...initial(), pod_requested: true };
  const rental = new Rental(
    api,
    state,
    () => {},
    () => 0,
  );
  await assert.rejects(rental.pod('public'), /unresolved/);
  await assert.rejects(rental.terminate(), /unresolved/);
  assert.equal(state.compute_terminated, undefined);
  assert.equal(api.calls.length, 0);
});
test('deadlines prevent creation; API errors retain storage and retryable termination', async () => {
  const api = new MockApi(),
    rental = new Rental(
      api,
      initial(),
      () => {},
      () => 2000000,
    );
  await assert.rejects(rental.volume(), /deadline/);
  await assert.rejects(rental.pod('public'), /deadline/);
  rental.now = () => 1;
  await rental.volume();
  assert.equal(api.calls.filter((c) => c.uri === '/pods').length, 0);
  api.resources = [{ id: 'pod1', name: rental.state.name }];
  rental.state.pod_id = 'pod1';
  api.deletionFails = true;
  await assert.rejects(rental.terminate(), /temporary/);
  assert.equal(rental.state.compute_terminated, undefined);
  await assert.rejects(rental.cleanupVolume(true), /Terminate compute/);
  api.deletionFails = false;
  await rental.terminate();
  await rental.cleanupVolume(false);
  assert.equal(api.volumes.length, 1);
  assert.equal(rental.state.volume_deleted, undefined);
  await rental.cleanupVolume(true);
  assert.equal(api.volumes.length, 0);
});
test('independent watchdog enforces setup and hard deadlines regardless of controller progress', () => {
  const state = initial();
  for (const now of [1, 1800000 - 1]) assert.equal(terminationDue(state, now), false);
  assert.equal(terminationDue(state, 1800000), true);
  state.setup_complete = true;
  assert.equal(terminationDue(state, 1800000), false);
  assert.equal(terminationDue(state, 7200000 - 1), false);
  assert.equal(terminationDue(state, 7200000), true);
  assert.equal(terminationDue(state, 100, true), true);
});
test('a definitive provisioning rejection cleans up the new empty volume', async () => {
  const api = new MockApi(),
    request = api.request.bind(api);
  api.request = async (method, uri, body) => {
    if (method === 'POST' && uri === '/pods') {
      const error = new Error('Insufficient balance');
      error.status = 400;
      throw error;
    }
    return request(method, uri, body);
  };
  const rental = new Rental(
    api,
    initial(),
    () => {},
    () => 100,
  );
  await rental.volume();
  await assert.rejects(rental.pod('public'), /balance/);
  assert.equal(rental.state.pod_rejected, true);
  await rental.terminate();
  await rental.cleanupVolume(false);
  assert.equal(api.volumes.length, 0);
  assert.equal(rental.state.compute_terminated, true);
});
test('lost volume response is reconciled and cleaned without another volume create', async () => {
  const api = new MockApi(),
    request = api.request.bind(api);
  api.request = async (method, uri, body) => {
    const result = await request(method, uri, body);
    if (method === 'POST' && uri === '/network-volumes') throw new Error('lost volume response');
    return result;
  };
  const rental = new Rental(
    api,
    initial(),
    () => {},
    () => 100,
  );
  await assert.rejects(rental.volume(), /lost volume/);
  await rental.terminate();
  await rental.cleanupVolume(false);
  assert.equal(api.volumes.length, 0);
  assert.equal(
    api.calls.filter((c) => c.method === 'POST' && c.uri === '/network-volumes').length,
    1,
  );
});
test('REST handles empty deletions, missing resources, errors and cursor pagination', async () => {
  const calls = [],
    api = new Api('test-key', async (url, options) => {
      calls.push({ url, options });
      if (options.method === 'DELETE') return new Response(null, { status: 204 });
      if (url.endsWith('/missing')) return new Response(null, { status: 404 });
      if (url.endsWith('/denied')) return new Response('secret provider detail', { status: 403 });
      return Response.json(
        url.includes('cursor=')
          ? { pods: [{ id: 'second' }], pagination: { nextCursor: null, hasNextPage: false } }
          : { pods: [{ id: 'first' }], pagination: { nextCursor: 'next page', hasNextPage: true } },
      );
    });
  assert.deepEqual(
    (await api.pods()).map((p) => p.id),
    ['first', 'second'],
  );
  assert.ok(calls[1].url.includes('cursor=next%20page'));
  assert.equal(await api.request('DELETE', '/pods/a'), null);
  assert.equal(await api.request('GET', '/missing'), null);
  await assert.rejects(
    api.request('GET', '/denied'),
    (e) => e.status === 403 && !e.message.includes('secret'),
  );
});
test('readiness accepts only finite checksummed v4 models at supported widths', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-readiness-model-'));
  try {
    const model = directory + '/model.blzn';
    for (const width of [64, 128, 256]) {
      modelFixture(model, 4, width);
      assert.deepEqual(readinessModel(model), {
        architecture: 4,
        width,
        sha256: checksum(fs.readFileSync(model)),
      });
    }
    modelFixture(model, 3);
    assert.throws(() => readinessModel(model), /v4 model/);
    modelFixture(model);
    const corrupt = fs.readFileSync(model);
    corrupt[30] ^= 1;
    fs.writeFileSync(model, corrupt);
    assert.throws(() => readinessModel(model), /checksum/);
    corrupt.writeFloatLE(Infinity, 24);
    fs.writeFileSync(
      model,
      Buffer.concat([corrupt.subarray(0, -64), Buffer.from(checksum(corrupt.subarray(0, -64)))]),
    );
    assert.throws(() => readinessModel(model), /nonfinite/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

function teacherArtifacts(args, changed = false, indexChanges = {}) {
  const output = args[1],
    option = (name) => args[args.indexOf(name) + 1];
  const states = 2 + Number(option('--previous-steps')),
    rows = states * 4;
  fs.mkdirSync(output, { recursive: true });
  const header = Buffer.alloc(12);
  header.write('BLZACT05');
  header.writeUInt32LE(4, 8);
  const vectors = [rows * 74, rows * 12, rows, rows, states + 1].map((count, i) => {
    const width = [2, 1, 4, 1, 4][i],
      bytes = Buffer.alloc(8 + count * width);
    bytes.writeBigUInt64LE(BigInt(count));
    if (i === 3 && changed) bytes[8] = 1;
    if (i === 4)
      for (let state = 0; state <= states; ++state) bytes.writeUInt32LE(state * 4, 8 + state * 4);
    return bytes;
  });
  const payload = Buffer.concat([header, ...vectors]);
  const contract = JSON.stringify({ batch: option('--candidate-batch') }),
    episode = 'identical completed mesh';
  fs.writeFileSync(output + '/actions.bin', payload);
  fs.writeFileSync(output + '/contract.json', contract);
  fs.writeFileSync(output + '/episode.bin', episode);
  fs.writeFileSync(
    output + '/index.json',
    JSON.stringify({
      schema: 4,
      complete: true,
      status: 'complete',
      training_started: false,
      requested_condition_available: true,
      reference_confirmed: true,
      preceding_lod_emitted: option('--previous-steps') === '1',
      states,
      queries: rows * 3,
      sha256: checksum(payload),
      contract_sha256: checksum(contract),
      episode_sha256: checksum(episode),
      timings: Object.fromEntries(
        [
          'load_and_session',
          'packing_and_baseline',
          'state_setup',
          'features_and_proposals',
          'candidate_build',
          'candidate_audit',
          'commit',
          'other_preparation',
        ].map((name) => [name + '_seconds', 0.01]),
      ),
      ...indexChanges,
    }),
  );
}

test('teacher readiness compares complete A/B/B/A payloads and preserves failures without training', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-readiness-'));
  const clean = { code: 0, signal: null, timed_out: false, cancelled: false, success: true };
  try {
    const model = directory + '/model.blzn';
    modelFixture(model);
    for (const failure of [
      'none',
      'payload',
      'predecessor',
      'states',
      'unknown',
      'timeout',
      'deadline',
    ]) {
      let clock = 1000,
        runs = 0;
      const batches = [],
        deadline = 601000;
      const execute = async (command, args, options) => {
        assert.ok(options.maximum > 0 && options.maximum <= deadline - clock);
        assert.equal(options.grace, 0);
        if (command === 'nvidia-smi') {
          fs.writeSync(
            options.stdio[1],
            '0, Fixture GPU, GPU-fixture, 580.0, 16384, 2048, 12, 1\n',
          );
          return clean;
        }
        assert.equal(command, 'build/neural/blitz-neural-placement-prepare');
        const option = (name) => args[args.indexOf(name) + 1];
        for (const [name, value] of Object.entries({
          '--architecture': '4',
          '--teacher-selection': 'policy-mixed',
          '--states': '2',
          '--pool': '4',
          '--pixels': '64',
          '--source-limit': '3',
          '--adjacent-limit': '2',
          '--gpu-memory-mib': '512',
          '--audit-mode': 'sparse',
          '--mask-only-coverage': 'on',
          '--training-profile': 'coverage',
          '--raster-backend': 'vulkan',
          '--vertex-storage': 'packed',
          '--model': model,
          '--seed': '101',
          '--corpus': readinessManifest,
          '--training-selection': readinessManifest,
        }))
          assert.equal(option(name), value);
        assert.ok(options.maximum <= 30000);
        const previous = option('--previous-steps') === '1';
        assert.equal(args.includes('--previous-pixels'), previous);
        if (previous) assert.equal(option('--previous-pixels'), '128');
        const batch = Number(option('--candidate-batch'));
        batches.push(batch);
        ++runs;
        const changes =
          failure === 'predecessor' && previous
            ? {
                requested_condition_available: false,
                preceding_lod_emitted: false,
                status: 'predecessor_unavailable',
              }
            : failure === 'states'
              ? { states: 1 }
              : failure === 'unknown'
                ? { status: 'unknown_audit', complete: false }
                : {};
        teacherArtifacts(args, failure === 'payload' && batch === 4, changes);
        clock += failure === 'deadline' ? deadline : batch === 1 ? 12000 : 6000;
        return failure === 'timeout' ? { ...clean, timed_out: true } : clean;
      };
      const report = await runTeacherReadiness({
        model,
        directory: directory + '/' + failure,
        execute,
        deadline,
        now: () => clock,
      });
      assert.equal(report.training_started, false);
      assert.equal(report.score, null);
      assert.equal(report.quality_proven, false);
      assert.equal(report.complete, failure === 'none');
      if (failure === 'none') {
        assert.equal(report.runs.length, 8);
        assert.deepEqual(batches, [1, 4, 4, 1, 1, 4, 4, 1]);
        assert.equal(report.speedup, 2);
        assert.equal(report.timing_authority, 'local_shared');
        assert.equal(report.gpu_before.available, true);
        assert.match(report.gpu_after.csv, /Fixture GPU/);
        for (const run of report.runs) assert.equal(run.rows, (2 + run.previous_steps) * 4);
      } else {
        assert.equal(report.speedup, null);
        assert.ok(report.comparisons.every((entry) => entry.speedup === null));
        assert.ok(report.error);
        if (failure === 'deadline') assert.equal(runs, 1);
      }
    }
    const expired = await runTeacherReadiness({
      model,
      directory: directory + '/expired',
      deadline: 1,
      now: () => 2,
      execute: () => assert.fail('expired readiness must not spawn'),
    });
    assert.equal(expired.complete, false);
    assert.equal(expired.runs.length, 0);
    const changed = await runTeacherReadiness({
      model,
      directory: directory + '/changed-model',
      expectedRequest: { model: { sha256: 'different-model' } },
      execute: () => assert.fail('changed immutable readiness inputs must not spawn'),
    });
    assert.equal(changed.complete, false);
    assert.match(changed.error, /request checksum mismatch/);
    assert.equal(changed.runs.length, 0);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('optional readiness shares the core test deadline and cannot turn incomplete work into success', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-core-readiness-'));
  try {
    const model = directory + '/model.blzn';
    modelFixture(model);
    let clock = 1000,
      teacherCalls = 0;
    const clean = { code: 0, signal: null, timed_out: false, cancelled: false, success: true };
    const report = await runCoreValidation({
      directory: directory + '/report',
      deadline: 601000,
      readinessModel: model,
      now: () => clock,
      execute: async (command, args, options) => {
        assert.ok(options.maximum <= 601000 - clock);
        if (command.includes('placement-prepare')) {
          assert.equal(options.maximum, 1000);
          ++teacherCalls;
          teacherArtifacts(args);
          clock = 601001;
        } else if (command === 'ctest') fs.writeSync(options.stdio[1], '100% tests passed\n');
        else if (command.endsWith('compute-sanitizer')) {
          fs.writeSync(options.stdio[1], 'ERROR SUMMARY: 0 errors\n');
          if (args.includes('build/neural/blitz-neural-vulkan-tests')) clock = 600000;
        }
        return clean;
      },
    });
    assert.equal(report.contracts_complete, true);
    assert.equal(report.complete, false);
    assert.equal(report.teacher_readiness.complete, false);
    assert.equal(report.teacher_readiness.speedup, null);
    assert.equal(report.training_started, false);
    assert.equal(teacherCalls, 1);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('core validation uses its separate cumulative grant and a 35 minute rental', () => {
  const now = 10_000_000,
    old = { experiment: 'gpu-refactor', name: 'previous-training', started_at: 0 };
  for (const id of coreValidationProfiles) {
    const profile = profiles[id],
      limits = deadlinesFor(now, undefined, profile);
    assert.equal(limits.setup_deadline_ms - now, 20 * 60000);
    assert.equal(limits.training_deadline_ms - limits.setup_deadline_ms, 10 * 60000);
    assert.equal(limits.deadline_ms - limits.training_deadline_ms, 5 * 60000);
    const budget = coreValidationBudget({ states: [old], rate: profile.gpu_hourly_usd_cap, now });
    assert.equal(budget.prior_assumed_usd, 0);
    assert.equal(budget.authorization.id, coreValidationAuthorization.id);
    assert.ok(budget.maximum_total_usd <= 1.25);
  }
  const prior = {
    name: 'core-fixture',
    experiment: 'core-validation',
    budget: { authorization: coreValidationAuthorization },
    deployment: { gpu_hourly_usd_cap: 0.3 },
    started_at: now - 30 * 60000,
    compute_terminated: true,
    terminated_at: now,
  };
  assert.equal(
    coreValidationBudget({ states: [old, prior, prior], rate: 0.3, now }).prior_assumed_usd,
    0.155,
  );
  assert.throws(
    () =>
      coreValidationBudget({
        states: [{ ...prior, started_at: now - 4 * 3600000 }],
        rate: 0.3,
        now,
      }),
    /grant/,
  );
  for (const state of [
    { ...prior, budget: {} },
    { ...prior, terminated_at: undefined },
    { ...prior, name: '' },
  ])
    assert.throws(() => coreValidationBudget({ states: [state], rate: 0.3, now }), /reconcile/);
  for (const rate of [0, -1, 1.11, NaN, Infinity])
    assert.throws(() => coreValidationBudget({ rate }));
  for (const minutes of [10, 30, 36, 180])
    assert.throws(() => coreValidationBudget({ rate: 0.3, minutes }));
});
test('core validation runs bounded contract and memory tests without a learning experiment', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-core-tests-'));
  const clean = { code: 0, signal: null, timed_out: false, cancelled: false, success: true };
  const expectedLog = (name) =>
    name === 'ctest'
      ? '100% tests passed, 0 tests failed out of 32\n'
      : name.endsWith('memcheck')
        ? '========= ERROR SUMMARY: 0 errors\n'
        : 'Vulkan contracts passed\n';
  try {
    let calls = 0,
      clock = 1000;
    const steps = coreValidationSteps(directory);
    assert.equal(steps.find((step) => step.name === 'vulkan-memcheck').args.at(-1), '--memcheck');
    assert.equal(
      steps.find((step) => step.name === 'action-gpu-memcheck').args.includes('--memcheck'),
      false,
    );
    const execute = async (command, args, options) => {
      const step = steps[calls++];
      assert.equal(command, step.command);
      assert.deepEqual(args, step.args);
      assert.ok(options.maximum > 0 && options.maximum <= 601000 - clock);
      assert.ok(!/teacher|cycle|train|prepare/.test([command, ...args].join(' ')));
      fs.writeSync(options.stdio[1], expectedLog(step.name));
      clock += 10000;
      return clean;
    };
    const report = await runCoreValidation({
      directory,
      deadline: 601000,
      execute,
      now: () => clock,
    });
    assert.equal(report.complete, true);
    assert.equal(calls, 4);
    assert.equal(report.training_started, false);
    assert.equal(report.score, null);
    assert.equal(report.quality_proven, false);
    assert.equal(JSON.parse(fs.readFileSync(directory + '/report.json')).complete, true);
    for (const bad of [
      { ...clean, code: 1 },
      { ...clean, signal: 'SIGTERM' },
      { ...clean, timed_out: true },
      { ...clean, cancelled: true },
    ]) {
      calls = 0;
      const result = await runCoreValidation({
        directory,
        deadline: 601000,
        now: () => 1000,
        execute: async (command, args, options) => {
          ++calls;
          fs.writeSync(options.stdio[1], expectedLog('ctest'));
          return bad;
        },
      });
      assert.equal(result.complete, false);
      assert.equal(calls, 1);
      assert.ok(result.error);
    }
    for (const deadline of [0, NaN]) {
      const result = await runCoreValidation({
        directory,
        deadline,
        now: () => 1000,
        execute: () => assert.fail('expired job must not execute'),
      });
      assert.equal(result.complete, false);
      assert.equal(result.phases.length, 0);
    }
    const controller = new AbortController();
    controller.abort();
    assert.equal(
      (
        await runCoreValidation({
          directory,
          deadline: 601000,
          signal: controller.signal,
          execute: () => assert.fail('cancelled job must not execute'),
        })
      ).complete,
      false,
    );
    calls = 0;
    clock = 1000;
    const late = await runCoreValidation({
      directory,
      deadline: 601000,
      now: () => clock,
      execute: async (command, args, options) => {
        fs.writeSync(options.stdio[1], expectedLog(steps[calls++].name));
        if (calls === 4) clock = 601001;
        return clean;
      },
    });
    assert.equal(late.complete, false);
    assert.equal(calls, 4);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
  for (const [name, log] of [
    ['ctest', '100% tests passed\n1: ***Skipped'],
    ['ctest', 'No tests were found'],
    ['vulkan-validation', 'VUID-invalid-command'],
    ['vulkan-validation', 'layer was not found'],
    ['action-gpu-memcheck', 'ERROR SUMMARY: 1 errors'],
    ['vulkan-memcheck', ''],
  ])
    assert.throws(() => validateCoreTestLog(name, log));
});
test('core preparation bundles only committed source and carries self-contained controller dependencies', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-core-bundle-')),
    root = directory + '/repo',
    output = root + '/runs/neural/rental';
  const cli = fileURLToPath(new URL('../scripts/neural/runpod.mjs', import.meta.url));
  const local = (command, args, options = {}) => {
    const result = spawnSync(command, args, { cwd: root, encoding: 'utf8', ...options });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout;
  };
  try {
    fs.mkdirSync(root + '/scripts/neural', { recursive: true });
    fs.writeFileSync(root + '/.gitignore', '/runs/\n/data/\n');
    const assets = readinessFixtures.map(({ asset }) => {
      const file = 'data/' + asset + '.gltf',
        bytes = Buffer.from(asset);
      fs.mkdirSync(root + '/data', { recursive: true });
      fs.writeFileSync(root + '/' + file, bytes);
      return { id: asset, split: 'development', files: [{ path: file, sha256: checksum(bytes) }] };
    });
    fs.mkdirSync(root + '/research/neural', { recursive: true });
    fs.writeFileSync(
      root + '/' + readinessManifest,
      JSON.stringify({ score_eligible: false, assets }),
    );
    for (const name of [
      'runpod.mjs',
      'runpod-api.mjs',
      'runpod-profile.mjs',
      'action-budget.mjs',
      'action-curriculum.mjs',
      'artifacts.mjs',
      'core-validation.mjs',
      'teacher-readiness.mjs',
      'bounded-process.mjs',
      'native-stack.mjs',
      'teacher-optimization.mjs',
      'quality-metrics.mjs',
    ])
      fs.copyFileSync(path.dirname(cli) + '/' + name, root + '/scripts/neural/' + name);
    local('git', ['init', '-q']);
    local('git', ['add', '.']);
    local('git', [
      '-c',
      'user.name=Fixture',
      '-c',
      'user.email=fixture@example.invalid',
      'commit',
      '-qm',
      'Deterministic controller fixture',
    ]);
    const env = { ...process.env };
    delete env.BLITZ_RUNPOD_PROFILE;
    delete env.BLITZ_CORE_READINESS_MODEL;
    delete env.BLITZ_CORE_STORAGE;
    local(process.execPath, [cli, 'prepare-core-validation', output], { env });
    const prepared = JSON.parse(fs.readFileSync(output + '/prepared.json'));
    assert.equal(prepared.experiment, 'core-validation');
    assert.equal(prepared.storage_mode, 'network');
    assert.equal(prepared.files, 1);
    assert.equal(prepared.deployment.id, 'hardware-validation-ada16');
    assert.equal(prepared.archive_name, 'input.tar.gz');
    assert.equal(
      checksum(fs.readFileSync(output + '/' + prepared.archive_name)),
      prepared.archive_sha256,
    );
    const archive = local('tar', ['-tf', output + '/' + inputArchive(prepared)]);
    assert.match(archive, /source.bundle/);
    assert.ok(!/dataset|assets|model[.]blzn/.test(archive));
    local(process.execPath, [output + '/control/runpod.mjs', 'status', output], { env });
    assert.equal(fs.existsSync(output + '/rental.json'), false);
    const model = directory + '/model.blzn',
      optional = root + '/runs/neural/optional';
    modelFixture(model);
    local(process.execPath, [cli, 'prepare-core-validation', optional], {
      env: { ...env, BLITZ_CORE_READINESS_MODEL: model, BLITZ_CORE_STORAGE: 'container' },
    });
    const optionalPrepared = JSON.parse(fs.readFileSync(optional + '/prepared.json'));
    assert.equal(optionalPrepared.storage_mode, 'container');
    assert.equal(optionalPrepared.teacher_readiness.model.sha256, checksum(fs.readFileSync(model)));
    assert.equal(optionalPrepared.files, 3 + assets.length);
    assert.equal(optionalPrepared.teacher_readiness.training_started, false);
    const extract = directory + '/extracted';
    fs.mkdirSync(extract);
    local('tar', ['-xf', optional + '/' + inputArchive(optionalPrepared), '-C', extract]);
    const inputs = JSON.parse(fs.readFileSync(extract + '/inputs.json'));
    assert.deepEqual(
      inputs.files.map((file) => file.path).sort(),
      [
        'source.bundle',
        'readiness/model.blzn',
        'readiness/request.json',
        ...assets.flatMap((asset) => asset.files.map((file) => 'assets/' + file.path)),
      ].sort(),
    );
    for (const file of inputs.files)
      assert.equal(file.sha256, checksum(fs.readFileSync(extract + '/' + file.path)));
    modelFixture(model, 3);
    const invalid = spawnSync(
      process.execPath,
      [cli, 'prepare-core-validation', root + '/runs/neural/invalid'],
      {
        cwd: root,
        encoding: 'utf8',
        env: { ...env, BLITZ_CORE_READINESS_MODEL: model },
      },
    );
    assert.equal(invalid.status, 1);
    assert.match(invalid.stderr, /v4 model/);
    assert.equal(fs.existsSync(root + '/runs/neural/invalid'), false);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
test('core prepare and launch reject ledger escapes before provider access', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-core-ledger-')),
    root = directory + '/repo',
    cli = fileURLToPath(new URL('../scripts/neural/runpod.mjs', import.meta.url)),
    guard = directory + '/deny-provider.mjs',
    marker = directory + '/provider-access';
  try {
    fs.mkdirSync(root + '/runs/neural', { recursive: true });
    const initialized = spawnSync('git', ['init', '-q'], { cwd: root, encoding: 'utf8' });
    assert.equal(initialized.status, 0, initialized.stderr);
    fs.writeFileSync(
      guard,
      `import fs from 'node:fs';globalThis.fetch=()=>{fs.writeFileSync(${JSON.stringify(marker)},'unexpected');throw new Error('unexpected provider access');};\n`,
    );
    const outside = directory + '/outside';
    fs.mkdirSync(outside);
    fs.symlinkSync(outside, root + '/runs/neural/linked');
    const paths = [outside, root + '/runs/neural/nested/rental', root + '/runs/neural/linked'];
    for (const rental of paths) {
      fs.mkdirSync(rental, { recursive: true });
      fs.writeFileSync(
        rental + '/prepared.json',
        JSON.stringify({ experiment: 'core-validation' }),
      );
      for (const command of ['prepare-core-validation', 'launch']) {
        const result = spawnSync(process.execPath, ['--import', guard, cli, command, rental], {
          cwd: root,
          env: { ...process.env, BLITZ_RUNPOD_PROFILE: 'hardware-validation-ada16' },
          encoding: 'utf8',
          timeout: 10000,
        });
        assert.equal(result.status, 1);
        assert.match(result.stderr, /canonical direct child of runs\/neural/);
        assert.equal(fs.existsSync(marker), false);
        assert.equal(fs.existsSync(rental + '/rental.json'), false);
      }
    }
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
test('container controller collects before termination and records ephemeral loss on failure', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-container-controller-'));
  const cli = fileURLToPath(new URL('../scripts/neural/runpod.mjs', import.meta.url));
  const preload = directory + '/mock-controller.mjs';
  try {
    fs.mkdirSync(directory + '/home/.config/blitz', { recursive: true });
    fs.writeFileSync(directory + '/home/.config/blitz/runpod-api-key', 'fixture-key', {
      mode: 0o600,
    });
    fs.writeFileSync(
      preload,
      `
import fs from 'node:fs';
import os from 'node:os';
import cp from 'node:child_process';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { syncBuiltinESMExports } from 'node:module';
import { createHash } from 'node:crypto';
const root = process.env.BLITZ_TEST_RENTAL;
const mode = process.env.BLITZ_TEST_OUTCOME;
const record = event => fs.appendFileSync(root + '/events.jsonl', JSON.stringify(event) + '\\n');
const hash = data => createHash('sha256').update(data).digest('hex');
const archive = Buffer.from('test-only collected evidence');
const state = JSON.parse(fs.readFileSync(root + '/rental.json'));
let pods = state.pod_id ? [{ id: state.pod_id, name: state.name }] : [];
os.homedir = () => process.env.BLITZ_TEST_HOME;
cp.execFileSync = (command, args) => {
  if (command !== 'systemctl') throw new Error('Unexpected local command: ' + command);
  record({ local: command, args }); return 'active';
};
cp.spawn = (command, args, options) => {
  if (command !== 'ssh') throw new Error('Unexpected subprocess: ' + command);
  const remote = args.at(-1); record({ ssh: remote });
  const child = new EventEmitter();
  child.stdin = new PassThrough(); child.stdin.resume();
  child.stdout = typeof options.stdio[1] === 'number' ? null : new PassThrough();
  child.stderr = new PassThrough(); child.kill = () => {};
  child.stdin.once('finish', () => {
    let data = '';
    if (remote === 'sha256sum /workspace/input.tar') {
      const prepared = JSON.parse(fs.readFileSync(root + '/prepared.json'));
      data = hash(fs.readFileSync(root + '/' + (prepared.archive_name ?? 'input.tar')));
    }
    else if (remote === 'cat /workspace/results.tar.gz.sha256') data = mode === 'collection-failure' ? 'invalid-checksum' : hash(archive);
    else if (remote === 'cat /workspace/results.tar.gz') data = archive;
    else if (remote.startsWith('if [ -f /workspace/job-finished ]') || remote.startsWith('test -f /workspace/job-finished')) data = 'finished';
    else if (remote !== 'true' && !remote.startsWith('mkdir -p /workspace && flock ') &&
             !remote.startsWith('cd /workspace && tar ') && !remote.startsWith('flock -o /workspace/launch.lock '))
      throw new Error('Unexpected remote command: ' + remote);
    if (child.stdout) child.stdout.emit('data', Buffer.from(data));
    else fs.writeSync(options.stdio[1], data);
    child.emit('close', 0, null);
  });
  return child;
};
syncBuiltinESMExports();
globalThis.fetch = async (url, options) => {
  const uri = new URL(url).pathname.replace('/v2', ''), method = options.method;
  record({ api: uri, method, body: options.body ? JSON.parse(options.body) : undefined });
  if (uri === '/pods' && method === 'GET') return Response.json({ pods, pagination: { nextCursor: null } });
  if (uri === '/pods' && method === 'POST') {
    const body = JSON.parse(options.body);
    if ('mounts' in body) throw new Error('Container Pod must omit mounts');
    const pod = { ...body, id: 'owned-pod' }; pods.push(pod); return Response.json(pod);
  }
  if (uri === '/pods/owned-pod' && method === 'DELETE') {
    pods = []; return new Response(null, { status: 204 });
  }
  if (uri === '/pods/owned-pod' && method === 'GET')
    return pods.length ? Response.json(pods[0]) : new Response(null, { status: 404 });
  throw new Error('Unexpected provider request: ' + method + ' ' + uri);
};
`,
    );
    for (const outcome of [
      'complete',
      'collection-failure',
      'deadline',
      'malformed-collection',
      'watchdog-malformed-collection',
      'missing-archive',
      'empty-archive',
    ]) {
      const run = directory + '/' + outcome;
      fs.mkdirSync(run);
      const now = Date.now(),
        profile = profiles['hardware-validation-ada16'];
      const state = {
        name: 'fixture-' + outcome,
        experiment: 'core-validation',
        storage_mode: 'container',
        deployment: profile,
        quote: { data_center: 'EU-1' },
        ...deadlinesFor(now, undefined, profile),
        endpoint: { host: 'fixture.invalid', username: 'root', port: 22 },
        ...(outcome === 'deadline' ||
        outcome.includes('malformed-collection') ||
        outcome.includes('archive')
          ? { deadline_ms: now - 120000, pod_requested: true, pod_id: 'owned-pod' }
          : {}),
      };
      fs.writeFileSync(run + '/rental.json', JSON.stringify(state));
      const archiveName = outcome === 'complete' ? 'input.tar.gz' : 'input.tar';
      fs.writeFileSync(run + '/' + archiveName, 'test-only input');
      fs.writeFileSync(run + '/identity.pub', 'public-fixture');
      if (outcome.includes('malformed-collection'))
        fs.writeFileSync(run + '/collection.json', '{bad-json');
      if (outcome.includes('archive'))
        fs.writeFileSync(
          run + '/collection.json',
          JSON.stringify({ verified: true, sha256: 'a'.repeat(64) }),
        );
      if (outcome === 'empty-archive') fs.writeFileSync(run + '/results.tar.gz', '');
      fs.writeFileSync(
        run + '/prepared.json',
        JSON.stringify({
          experiment: 'core-validation',
          storage_mode: 'container',
          revision: 'fixture',
          ...(outcome === 'complete' ? { archive_name: archiveName } : {}),
          archive_sha256: checksum('test-only input'),
        }),
      );
      const result = spawnSync(
        process.execPath,
        ['--import', preload, cli, outcome.startsWith('watchdog') ? 'watchdog' : 'control', run],
        {
          encoding: 'utf8',
          timeout: 10000,
          env: {
            ...process.env,
            BLITZ_RUNPOD_PROFILE: profile.id,
            BLITZ_TEST_RENTAL: run,
            BLITZ_TEST_OUTCOME: outcome,
            BLITZ_TEST_HOME: directory + '/home',
          },
        },
      );
      assert.equal(result.status, 0, result.stderr);
      const ended = JSON.parse(
        fs.readFileSync(run + (outcome.startsWith('watchdog') ? '/watchdog.json' : '/rental.json')),
      );
      assert.equal(ended.storage_mode, 'container');
      assert.equal(ended.compute_terminated, true);
      assert.equal(ended.ephemeral_storage_lost, outcome !== 'complete');
      assert.equal(ended.results_verified, outcome === 'complete');
      if (!outcome.startsWith('watchdog'))
        assert.equal(
          ended.phase,
          outcome === 'complete' ? 'complete' : 'stopped_uncollected_ephemeral',
        );
      const events = fs
        .readFileSync(run + '/events.jsonl', 'utf8')
        .trim()
        .split('\n')
        .map(JSON.parse);
      assert.ok(events.every((event) => !event.api?.includes('network-volumes')));
      if (!outcome.startsWith('watchdog'))
        assert.equal(JSON.parse(result.stdout).volume_preserved, false);
      if (outcome === 'complete') {
        assert.equal(
          fs.readFileSync(run + '/results.tar.gz', 'utf8'),
          'test-only collected evidence',
        );
        const collected = events.findIndex(
          (event) => event.ssh === 'cat /workspace/results.tar.gz',
        );
        const deleted = events.findIndex((event) => event.method === 'DELETE');
        assert.ok(collected >= 0 && deleted > collected);
        assert.ok(events.some((event) => event.ssh?.startsWith('mkdir -p /workspace && flock ')));
      } else
        assert.equal(
          fs.existsSync(run + '/collection.json'),
          outcome.includes('malformed-collection') || outcome.includes('archive'),
        );
    }
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('malformed storage cannot launch or prepare training, and cannot block a stop request', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-storage-guards-'));
  const cli = fileURLToPath(new URL('../scripts/neural/runpod.mjs', import.meta.url));
  try {
    for (const state of [
      { experiment: 'gpu-refactor', storage_mode: 'container' },
      { experiment: 'core-validation', storage_mode: 'typo' },
      { experiment: 'core-validation', storage_mode: 'container', volume_id: 'existing' },
    ]) {
      fs.writeFileSync(directory + '/prepared.json', JSON.stringify(state));
      const launch = spawnSync(process.execPath, [cli, 'launch', directory], { encoding: 'utf8' });
      assert.equal(launch.status, 1);
      assert.match(launch.stderr, /storage|network-volume/);
      assert.equal(fs.existsSync(directory + '/rental.json'), false);
    }
    const prepare = spawnSync(process.execPath, [cli, 'prepare-gpu-refactor', directory], {
      encoding: 'utf8',
      env: { ...process.env, BLITZ_CORE_STORAGE: 'container' },
    });
    assert.equal(prepare.status, 1);
    assert.match(prepare.stderr, /requires core-validation/);
    fs.writeFileSync(
      directory + '/rental.json',
      JSON.stringify({ storage_mode: 'typo', pod_id: 'owned' }),
    );
    const stopped = spawnSync(process.execPath, [cli, 'stop', directory], { encoding: 'utf8' });
    assert.equal(stopped.status, 0, stopped.stderr);
    assert.equal(fs.existsSync(directory + '/stop-requested'), true);
    fs.rmSync(directory + '/stop-requested');
    fs.writeFileSync(directory + '/rental.json', '{bad-json');
    const malformed = spawnSync(process.execPath, [cli, 'stop', directory], { encoding: 'utf8' });
    assert.equal(malformed.status, 0, malformed.stderr);
    assert.equal(fs.existsSync(directory + '/stop-requested'), true);
    assert.match(malformed.stdout, /metadata is unavailable/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
test('core test rental terminates only its resources and keeps unverified evidence', async () => {
  const api = new MockApi(),
    profile = profiles['hardware-validation-ada16'];
  const state = {
    ...initial(),
    name: 'core-validation-fixture',
    experiment: 'core-validation',
    deployment: profile,
    ...deadlinesFor(100, undefined, profile),
    budget: coreValidationBudget({ rate: profile.gpu_hourly_usd_cap, now: 100 }),
  };
  const rental = new Rental(
    api,
    state,
    () => {},
    () => 200,
  );
  await rental.volume();
  await rental.pod('public');
  api.resources.push({ id: 'unrelated', name: 'current-training' });
  await rental.terminate();
  await rental.cleanupVolume(false);
  assert.equal(state.compute_terminated, true);
  assert.equal(api.volumes.length, 1);
  assert.deepEqual(
    api.resources.map((p) => p.id),
    ['unrelated'],
  );
  await rental.cleanupVolume(true);
  assert.equal(state.volume_deleted, true);
  assert.equal(api.volumes.length, 0);
});
