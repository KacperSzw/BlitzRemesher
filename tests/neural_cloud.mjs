import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
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
    fs.writeFileSync(root + '/.gitignore', '/runs/\n');
    for (const name of [
      'runpod.mjs',
      'runpod-api.mjs',
      'runpod-profile.mjs',
      'action-budget.mjs',
      'action-curriculum.mjs',
      'artifacts.mjs',
      'core-validation.mjs',
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
    local(process.execPath, [cli, 'prepare-core-validation', output], { env });
    const prepared = JSON.parse(fs.readFileSync(output + '/prepared.json'));
    assert.equal(prepared.experiment, 'core-validation');
    assert.equal(prepared.files, 1);
    assert.equal(prepared.deployment.id, 'hardware-validation-ada16');
    const archive = local('tar', ['-tf', output + '/input.tar']);
    assert.match(archive, /source.bundle/);
    assert.ok(!/dataset|assets|model[.]blzn/.test(archive));
    local(process.execPath, [output + '/control/runpod.mjs', 'status', output], { env });
    assert.equal(fs.existsSync(output + '/rental.json'), false);
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
