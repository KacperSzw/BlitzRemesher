import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { runTeacherOptimization } from '../scripts/neural/teacher-optimization-job.mjs';
import { selectNvidiaIcd, verifyNvidiaIcdSelection } from '../scripts/neural/nvidia-icd.mjs';
import {
  teacherOptimizationAuthorization,
  teacherOptimizationBudget,
  teacherOptimizationProfiles,
  validateOptimizationRequest,
  optimizationCycleArguments,
  teacherStrategyGate,
  validateOptimizationPilot,
  validateOptimizationCheckpoint,
  selectOptimizationAssets,
} from '../scripts/neural/teacher-optimization.mjs';
import { profiles } from '../scripts/neural/runpod-profile.mjs';
import { rentalDeadlines, storageMode } from '../scripts/neural/runpod-api.mjs';

const request = () => ({
  version: 1,
  final_training: false,
  baseline_revision: 'a'.repeat(40),
  initialization_sha256: 'b'.repeat(64),
  seeds: [101, 211, 307],
  action_trials: 512,
  action_batch: 16,
  gpu_memory_mib: 16384,
  workers: 2,
  candidate_batch: 4,
  learning_minutes: 5,
  finalize_minutes: 2,
});

test('audit-only development assets are bundled without admitting them into teachers', () => {
  const options = {
    assets: [
      { id: 'train', split: 'development' },
      { id: 'audit', split: 'development' },
      { id: 'held-out', split: 'validation' },
    ],
    trainingIds: ['train'],
    teacherIds: ['train'],
    auditIds: ['train', 'audit'],
  };
  assert.deepEqual(
    selectOptimizationAssets(options).map((a) => a.id),
    ['train', 'audit'],
  );
  assert.throws(
    () => selectOptimizationAssets({ ...options, teacherIds: ['audit'] }),
    /training split/,
  );
  assert.throws(
    () => selectOptimizationAssets({ ...options, auditIds: ['held-out'] }),
    /development assets/,
  );
  assert.throws(() => selectOptimizationAssets({ ...options, auditIds: ['missing'] }), /Missing/);
});

test('optimization grant counts failed and active rentals and cannot extend core deadlines', () => {
  const now = 10_000_000;
  for (const id of teacherOptimizationProfiles) {
    const profile = profiles[id],
      deadlines = rentalDeadlines(now, undefined, profile);
    assert.equal(deadlines.deadline_ms - now, 140 * 60000);
    assert.equal(deadlines.setup_deadline_ms - now, 20 * 60000);
    assert.equal(deadlines.deadline_ms - deadlines.training_deadline_ms, 8 * 60000);
    assert.ok(profile.catalog_vram_gb >= 24);
    assert.ok(
      teacherOptimizationBudget({ rate: profile.gpu_hourly_usd_cap, now }).maximum_total_usd <= 3,
    );
  }
  assert.equal(
    rentalDeadlines(now, undefined, profiles['hardware-validation-ada16']).deadline_ms - now,
    35 * 60000,
  );
  const state = {
    name: 'failed-setup',
    experiment: 'teacher-optimization',
    budget: { authorization: teacherOptimizationAuthorization },
    deployment: { gpu_hourly_usd_cap: 0.55 },
    started_at: now - 30 * 60000,
    compute_terminated: true,
    terminated_at: now,
  };
  const settled = teacherOptimizationBudget({ states: [state, state], rate: 0.55, now });
  assert.ok(Math.abs(settled.prior_assumed_usd - 0.29) < 1e-12);
  const active = teacherOptimizationBudget({
    states: [{ ...state, compute_terminated: false }],
    rate: 0.55,
    now: now + 60000,
  });
  assert.ok(active.prior_assumed_usd > settled.prior_assumed_usd);
  assert.throws(
    () =>
      teacherOptimizationBudget({
        states: [{ ...state, started_at: now - 6 * 3600000 }],
        rate: 0.55,
        now,
      }),
    /grant/,
  );
  for (const broken of [
    { ...state, budget: {} },
    { ...state, started_at: NaN },
    { ...state, terminated_at: undefined },
  ])
    assert.throws(
      () => teacherOptimizationBudget({ states: [broken], rate: 0.55, now }),
      /reconcile/,
    );
  for (const rate of [0, -1, 1.11, NaN, Infinity])
    assert.throws(() => teacherOptimizationBudget({ rate }));
  assert.throws(() => teacherOptimizationBudget({ rate: 0.55, minutes: 141 }));
});

test('disposable storage remains limited to the two bounded experiments', () => {
  assert.equal(
    storageMode({ experiment: 'teacher-optimization', storage_mode: 'container' }),
    'container',
  );
  for (const extra of [
    { volume_id: 'retained' },
    { volume_requested: true },
    { experiment: 'gpu-refactor' },
  ])
    assert.throws(() =>
      storageMode({ experiment: 'teacher-optimization', storage_mode: 'container', ...extra }),
    );
});

test('pilot commands preserve initializer, learning limits and explicit strategy', () => {
  for (const strategy of ['exhaustive', 'coverage-core-first']) {
    const args = optimizationCycleArguments(request(), {
      run: 'run',
      seed: 211,
      strategy,
      model: 'initial.blzn',
      curriculum: 'conditions.json',
    });
    const option = (key) => args[args.indexOf(key) + 1];
    assert.equal(option('--initialize'), 'initial.blzn');
    assert.equal(option('--teacher-strategy'), strategy);
    assert.equal(option('--seed'), '211');
    assert.equal(option('--duration-minutes'), '5');
    assert.equal(option('--finalize-minutes'), '2');
    assert.equal(option('--states'), '16');
    assert.equal(option('--updates'), '128');
  }
  for (const patch of [
    { final_training: true },
    { baseline_revision: 'HEAD' },
    { learning_minutes: 6 },
    { seeds: [101] },
    { gpu_memory_mib: 32768 },
  ])
    assert.throws(() => validateOptimizationRequest({ ...request(), ...patch }));
  assert.throws(() => optimizationCycleArguments(request(), { seed: 99, strategy: 'exhaustive' }));
});

function audit(delta = -1) {
  const assets = Array.from({ length: 12 }, (_, i) => ({
    id: 'asset' + i,
    category: ['manufactured', 'organic', 'rocks', 'stress'][i % 4],
  }));
  return {
    version: 2,
    complete: true,
    manifest_sha256: 'manifest',
    visual_settings_sha256: 'visual',
    execution_settings_sha256: 'execution',
    expected_assets: assets,
    execution_settings: { audit_rankings: ['learned'] },
    rows: assets.map((a) => ({
      asset: a.id,
      category: a.category,
      ranking: 'learned',
      status: 'complete',
      seconds: 1,
      neural: {
        action_diagnostics_version: 1,
        action_proposals: [{ stop_reason: 'no_accepted_action' }],
      },
      lods: Array.from({ length: 8 }, (_, i) => ({
        triangles: i ? 1000 - i * 50 + delta : 1000,
        source: { complete: true, passed: true },
        adjacent: { complete: true, passed: true },
      })),
    })),
  };
}
test('a single per-LOD regression cannot be hidden by aggregate gains or partial quality evidence', () => {
  const pairs = [101, 211, 307].map((seed) => ({
    seed,
    exhaustive: audit(0),
    candidate: audit(-2),
  }));
  assert.equal(teacherStrategyGate(pairs, 12).passed, true);
  pairs[0].candidate.rows[0].neural.action_proposals[0].stop_reason = 'trial_budget';
  const censored = teacherStrategyGate(pairs, 12);
  assert.equal(censored.matched_budget_nonregression, true);
  assert.equal(censored.passed, false);
  assert.equal(censored.comparisons[0].termination[1].uncensored, false);
  pairs[0].candidate.rows[0].neural.action_proposals[0].stop_reason = 'no_accepted_action';
  pairs[1].candidate.rows[0].lods[2].triangles = pairs[1].exhaustive.rows[0].lods[2].triangles + 1;
  assert.equal(teacherStrategyGate(pairs, 12).passed, false);
  assert.throws(() => teacherStrategyGate(pairs.slice(1), 12));
  assert.throws(() => teacherStrategyGate(pairs, 4));
  pairs[1].candidate.complete = false;
  assert.throws(() => teacherStrategyGate(pairs, 12));
});

function experiment(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-teacher-optimization-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const model = directory + '/model.blzn';
  fs.writeFileSync(model, 'test-owned initializer');
  const teardownBinary = directory + '/teardown-fixture';
  fs.writeFileSync(teardownBinary, 'test-owned native fixture identity');
  return {
    directory: directory + '/results',
    model,
    teardownBinary,
    request: {
      ...request(),
      initialization_sha256: createHash('sha256').update(fs.readFileSync(model)).digest('hex'),
    },
    deadline: 112 * 60000,
    now: () => 0,
  };
}

function icdFixture(t, backend = 'glx') {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-nvidia-icd-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const library = directory + (backend === 'glx' ? '/libGLX_nvidia.so.0' : '/libEGL_nvidia.so.0');
  fs.writeFileSync(library, 'test-owned library identity');
  const original = JSON.stringify({
    file_format_version: '1.0.0',
    ICD: { api_version: '1.4.303', library_path: library },
  });
  const selected = selectNvidiaIcd(original, {
    sourcePath: directory + '/injected.json',
    selectedPath: directory + '/selected.json',
    requested: backend,
  });
  const provenance = {
    ...selected.provenance,
    original_copy_path: directory + '/original.json',
    dependency_library_resolution: 'manifest_path',
    dependency_library_path: library,
    dependency_library_sha256: createHash('sha256').update(fs.readFileSync(library)).digest('hex'),
  };
  fs.writeFileSync(provenance.original_copy_path, original);
  fs.writeFileSync(provenance.selected_path, selected.contents);
  const file = directory + '/selection.json';
  fs.writeFileSync(file, JSON.stringify(provenance));
  return { file, provenance, environment: { VK_DRIVER_FILES: provenance.selected_path } };
}

test('vendor ICD selection preserves GLX/EGL manifests, paths, API versions and extra fields', () => {
  for (const backend of ['GLX', 'EGL'])
    for (const prefix of ['', '/opt/driver/lib/'])
      for (const api of ['1.3.289', '1.4.303']) {
        const library = `${prefix}lib${backend}_nvidia.so.0`;
        const manifest = {
          file_format_version: '1.0.1',
          ICD: { library_path: library, api_version: api, is_portability_driver: false },
          vendor_metadata: { retained: true },
        };
        const contents = JSON.stringify(manifest, null, 4) + '\n';
        const selected = selectNvidiaIcd(contents, {
          sourcePath: '/etc/vulkan/icd.d/nvidia_icd.json',
          selectedPath: '/workspace/results/nvidia-selected.json',
        });
        assert.equal(selected.contents, contents);
        assert.equal(selected.provenance.backend, backend.toLowerCase());
        assert.equal(selected.provenance.api_version, api);
        assert.equal(selected.provenance.source_sha256, selected.provenance.selected_sha256);
        assert.equal(selected.provenance.selected_library_path, library);
        assert.deepEqual(JSON.parse(selected.contents), manifest);
      }
});

test('copying a directory-relative ICD retains the original library resolution', () => {
  const manifest = {
    file_format_version: '1.0.0',
    ICD: { library_path: '../../../lib/libGLX_nvidia.so.0', api_version: '1.4.303' },
  };
  const selected = selectNvidiaIcd(JSON.stringify(manifest), {
    sourcePath: '/opt/driver/share/vulkan/icd.d/nvidia_icd.json',
    selectedPath: '/workspace/results/selected.json',
    requested: 'glx',
  });
  assert.equal(selected.provenance.selected_library_path, '/opt/driver/lib/libGLX_nvidia.so.0');
  assert.equal(selected.provenance.source_library_path, manifest.ICD.library_path);
  assert.deepEqual(JSON.parse(selected.contents), {
    ...manifest,
    ICD: { ...manifest.ICD, library_path: '/opt/driver/lib/libGLX_nvidia.so.0' },
  });
  assert.notEqual(selected.provenance.selected_sha256, selected.provenance.source_sha256);
});

test('ICD selection rejects unsupported entries and explicit backend mismatches without rewriting', () => {
  const options = { sourcePath: '/etc/nvidia.json', selectedPath: '/results/selected.json' };
  const manifest = (library) =>
    JSON.stringify({
      file_format_version: '1.0.0',
      ICD: { library_path: library, api_version: '1.4.303' },
    });
  for (const library of ['libvulkan_intel.so', 'libGLX.so.0', 'libGLX_nvidia.so.1', '', null])
    assert.throws(() => selectNvidiaIcd(manifest(library), options), /ICD/);
  for (const [requested, vendor] of [
    ['glx', 'EGL'],
    ['egl', 'GLX'],
  ])
    assert.throws(
      () => selectNvidiaIcd(manifest(`lib${vendor}_nvidia.so.0`), { ...options, requested }),
      /vendor manifest selects/,
    );
  for (const requested of ['auto', '', null])
    assert.throws(
      () => selectNvidiaIcd(manifest('libGLX_nvidia.so.0'), { ...options, requested }),
      /selection/,
    );
});

test('ICD verification checks selected manifest and driver library hashes', (t) => {
  const fixture = icdFixture(t);
  assert.deepEqual(verifyNvidiaIcdSelection(fixture.file, 'glx'), fixture.provenance);
  assert.throws(() => verifyNvidiaIcdSelection(fixture.file, 'egl'), /vendor manifest selects/);
  const original = fs.readFileSync(fixture.provenance.selected_path);
  fs.appendFileSync(fixture.provenance.selected_path, '\n');
  assert.throws(() => verifyNvidiaIcdSelection(fixture.file, 'glx'), /ICD checksum/);
  fs.writeFileSync(fixture.provenance.selected_path, original);
  fs.appendFileSync(fixture.provenance.dependency_library_path, 'changed');
  assert.throws(() => verifyNvidiaIcdSelection(fixture.file, 'glx'), /library checksum/);
});

test('frozen optimization ICD choice is explicit and rejects unknown selections', () => {
  for (const vulkan_icd of [undefined, 'glx', 'egl'])
    assert.equal(validateOptimizationRequest({ ...request(), vulkan_icd }).vulkan_icd, vulkan_icd);
  for (const vulkan_icd of ['vendor', 'auto', false, null])
    assert.throws(
      () => validateOptimizationRequest({ ...request(), vulkan_icd }),
      /invalid frozen/,
    );
});

test('an explicit ICD request requires matching provenance and a clean loader environment', async (t) => {
  for (const invalid of ['missing', 'wrong-backend', 'wrong-path', 'legacy-override']) {
    const options = experiment(t),
      fixture = icdFixture(t);
    options.request.vulkan_icd = invalid === 'wrong-backend' ? 'egl' : 'glx';
    const environment = { ...fixture.environment };
    if (invalid === 'wrong-path') environment.VK_DRIVER_FILES = '/other/manifest.json';
    if (invalid === 'legacy-override') environment.VK_ICD_FILENAMES = '/stale/manifest.json';
    const report = await runTeacherOptimization({
      ...options,
      icdSelection: invalid === 'missing' ? undefined : fixture.file,
      environment,
      contracts: async () => assert.fail('ICD mismatch must stop native work'),
      execute: async () => assert.fail('ICD mismatch must stop native work'),
    });
    assert.equal(report.complete, false);
    assert.equal(report.training_started, false);
    assert.match(report.error, /ICD|Vulkan/);
  }
});

test('timeout diagnostics require an explicit boolean request', () => {
  for (const flag of [undefined, false, true])
    assert.equal(
      validateOptimizationRequest({ ...request(), timeout_diagnostics: flag }).timeout_diagnostics,
      flag,
    );
  for (const flag of [1, 'true', {}, null])
    assert.throws(
      () => validateOptimizationRequest({ ...request(), timeout_diagnostics: flag }),
      /invalid frozen/,
    );
});

test('failed native attach preflight blocks GPU stress, comparisons and learning', async (t) => {
  const options = experiment(t);
  options.request.timeout_diagnostics = true;
  const report = await runTeacherOptimization({
    ...options,
    debuggerPreflight: async (config) => {
      assert.equal(config.binary, 'build/neural/blitz-neural-placement-prepare');
      assert.equal(config.debuggerCommand, 'gdb');
      assert.ok(config.deadline <= 10000);
      return { complete: false, process: { diagnostic: { captured: false, ptrace_denied: true } } };
    },
    contracts: async () => assert.fail('no GPU contracts before successful attach'),
    profile: async () => assert.fail('no benchmark before successful attach'),
    execute: async () => assert.fail('no GPU stress or learning before successful attach'),
  });
  assert.equal(report.complete, false);
  assert.equal(report.training_started, false);
  assert.match(report.error, /preflight failed/);
});

test('diagnostic retry bounds teardown stress and forwards capture to paired cycles', async (t) => {
  const options = experiment(t);
  options.request.timeout_diagnostics = true;
  options.request.vulkan_icd = 'glx';
  const fixture = icdFixture(t);
  const calls = [];
  let clock = 0;
  const report = await runTeacherOptimization({
    ...options,
    icdSelection: fixture.file,
    environment: fixture.environment,
    now: () => clock,
    debuggerPreflight: async () => {
      clock += 6000;
      return { complete: true };
    },
    contracts: async (config) => {
      assert.equal(calls.length, 4);
      assert.equal(config.deadline - clock, 5 * 60000);
      assert.ok(config.deadline <= 20 * 60000);
      await config.execute('contract-fixture', [], {
        env: { VK_DRIVER_FILES: '/wrong.json', VK_ICD_FILENAMES: '/legacy.json' },
      });
      return { complete: true };
    },
    profile: async (config) => {
      assert.deepEqual(config.timeoutDiagnostics, { debuggerCommand: 'gdb', maximum: 5000 });
      assert.equal(config.deadline, 20 * 60000);
      await config.execute('profile-fixture', [], {});
      return { complete: true };
    },
    execute: async (command, args, config) => {
      assert.equal(config.env.VK_DRIVER_FILES, fixture.provenance.selected_path);
      assert.equal(config.env.VK_ICD_FILENAMES, undefined);
      if (command === 'contract-fixture' || command === 'profile-fixture')
        return { success: true, code: 0, signal: null };
      calls.push(command);
      assert.equal(config.grace, 6000);
      assert.equal(config.env.BLITZ_ALLOW_DEBUGGER_ATTACH, '1');
      assert.equal(config.env.VK_DRIVER_FILES, fixture.provenance.selected_path);
      assert.equal(config.env.VK_ICD_FILENAMES, undefined);
      assert.equal(config.timeoutDiagnostic.maximum, 5000);
      if (calls.length <= 4) {
        assert.equal(command, options.teardownBinary);
        assert.deepEqual(args, ['--teardown']);
        assert.equal(config.maximum, 45000);
        const saved = JSON.parse(fs.readFileSync(options.directory + '/report.json'));
        assert.equal(
          saved.teardown_stress.binary_sha256,
          createHash('sha256').update(fs.readFileSync(command)).digest('hex'),
        );
        assert.equal(saved.teardown_stress.requested_rounds, 48);
        assert.equal(saved.teardown_stress.completed_runs, calls.length - 1);
        assert.equal(saved.vulkan_icd.selected_sha256, fixture.provenance.selected_sha256);
        assert.equal(saved.vulkan_icd.VK_DRIVER_FILES, config.env.VK_DRIVER_FILES);
        clock += 45000;
        return { success: true, code: 0, signal: null };
      }
      assert.equal(command, 'build/neural/blitz-neural-cycle');
      assert.equal(config.maximum + config.grace, 7 * 60000);
      return {
        success: false,
        code: null,
        signal: 'SIGKILL',
        timed_out: true,
        diagnostic: { captured: true },
      };
    },
  });
  assert.equal(calls.length, 5);
  assert.equal(report.complete, false);
  assert.equal(report.training_started, true);
  assert.equal(report.teardown_stress.completed_full_run_rounds, 48);
  assert.equal(report.phases.at(-1).diagnostic.captured, true);
});

test('failed stress stops remaining runs and short deadlines cannot extend the teacher phase', async (t) => {
  for (const failure of ['stress', 'deadline']) {
    const options = experiment(t);
    options.request.timeout_diagnostics = true;
    let clock = 0,
      calls = 0;
    const report = await runTeacherOptimization({
      ...options,
      deadline: failure === 'deadline' ? 51000 : options.deadline,
      now: () => clock,
      debuggerPreflight: async () => ({ complete: true }),
      contracts: async () => assert.fail('stress failure must stop contracts'),
      profile: async () => assert.fail('stress failure must stop profiling'),
      execute: async (_command, _args, config) => {
        calls++;
        assert.ok(clock + config.maximum + config.grace <= Math.min(options.deadline, 20 * 60000));
        if (failure === 'stress')
          return { success: false, code: null, signal: 'SIGKILL', timed_out: true };
        clock += 45000;
        return { success: true, code: 0, signal: null };
      },
    });
    assert.equal(calls, 1);
    assert.equal(report.complete, false);
    assert.equal(report.training_started, false);
    assert.equal(report.teardown_stress.completed_runs, failure === 'deadline' ? 1 : 0);
    assert.match(
      report.error,
      failure === 'stress' ? /teardown-stress-1 failed/ : /insufficient deadline/,
    );
  }
});

test('invalid engineering evidence prevents profiles and learning while preserving failure artifacts', async (t) => {
  const options = experiment(t);
  const report = await runTeacherOptimization({
    ...options,
    contracts: async () => ({ complete: false }),
    profile: async () => assert.fail('profile must not run'),
    execute: async () => assert.fail('learning must not run'),
  });
  assert.equal(report.complete, false);
  assert.equal(report.training_started, false);
  assert.equal(report.strategy_promotable, false);
  assert.equal(report.score, null);
  assert.match(report.error, /engineering contracts/);
  assert.deepEqual(JSON.parse(fs.readFileSync(options.directory + '/report.json')), report);
});

test('both teacher comparisons use the frozen device budget and failure blocks learning', async (t) => {
  const options = experiment(t),
    calls = [];
  const report = await runTeacherOptimization({
    ...options,
    contracts: async () => ({ complete: true }),
    profile: async (p) => {
      calls.push(p);
      return { complete: calls.length === 1 };
    },
    execute: async () => assert.fail('incomplete profile must prevent learning'),
  });
  assert.equal(calls.length, 2);
  for (const p of calls) {
    assert.equal(p.gpuMemoryMiB, options.request.gpu_memory_mib);
    assert.equal(p.workers, options.request.workers);
    assert.equal(p.candidateBatch, options.request.candidate_batch);
    assert.equal(p.timingAuthority, 'isolated_remote');
  }
  assert.equal(calls[1].comparison, 'strategy');
  assert.equal(calls[1].baseline, calls[1].optimized);
  assert.equal(report.complete, false);
  assert.equal(report.training_started, false);
  assert.equal(report.strategy_promotable, false);
});

test('the runner reserves a complete pair and preserves signalled pilot failure', async (t) => {
  for (const [remaining, runPilot] of [
    [13, false],
    [14, true],
  ]) {
    const options = experiment(t);
    let now = 0,
      calls = 0;
    const report = await runTeacherOptimization({
      ...options,
      now: () => now,
      contracts: async () => ({ complete: true }),
      profile: async () => {
        now = (62 - remaining) * 60000;
        return { complete: true };
      },
      execute: async (_command, _args, config) => {
        calls++;
        assert.equal(config.maximum, 7 * 60000);
        return { success: true, code: 0, signal: 'SIGTERM' };
      },
    });
    assert.equal(calls, runPilot ? 1 : 0);
    assert.equal(report.training_started, runPilot);
    assert.equal(report.complete, false);
    assert.equal(report.strategy_promotable, false);
    assert.match(report.error, runPilot ? /SIGTERM/ : /complete paired pilot/);
  }
});

test('pilot evidence must identify its seed, strategy, initializer and fresh optimizer updates', () => {
  const options = {
    request: request(),
    seed: 211,
    strategy: 'coverage-core-first',
    hashes: { binary_sha256: 'binary' },
    contract: {
      version: 9,
      architecture: 4,
      hidden_width: 64,
      seed: 211,
      teacher_strategy: 'coverage-core-first',
      initialize: 'b'.repeat(64),
      warmstart: '',
      duration_minutes: 5,
      finalize_minutes: 2,
      states: 16,
      updates: 128,
      batch: 512,
      workers: 2,
      candidate_batch: 4,
      gpu_memory_mib: 16384,
      training_profile: 'coverage',
      mask_only_coverage: true,
      raster: 'vulkan-v1',
      storage: 'packed',
      policy_action_candidates: true,
      policy_rollout_trials: 64,
      episode_seeds: true,
      simplifier_seeds: true,
      quality: false,
      update_backend: 'fp32-compensated-v2',
      binary_sha256: 'binary',
    },
    result: {
      complete: true,
      updates_this_invocation: 64,
      failed_conditions_count: 0,
      checkpoint: 'step-64',
      checkpoint_sha256: 'c'.repeat(64),
    },
    latest: { complete: true, checkpoint: 'step-64', checkpoint_sha256: 'c'.repeat(64) },
  };
  assert.doesNotThrow(() => validateOptimizationPilot(options));
  for (const patch of [
    { seed: 101 },
    { teacher_strategy: 'exhaustive' },
    { initialize: 'd'.repeat(64) },
    { binary_sha256: 'stale' },
  ])
    assert.throws(
      () => validateOptimizationPilot({ ...options, contract: { ...options.contract, ...patch } }),
      /contract mismatch/,
    );
  for (const patch of [
    { complete: false },
    { updates_this_invocation: 0 },
    { updates_this_invocation: NaN },
    { failed_conditions_count: 1 },
    { checkpoint: 'step-1' },
  ])
    assert.throws(
      () => validateOptimizationPilot({ ...options, result: { ...options.result, ...patch } }),
      /incomplete, stale or numerically invalid/,
    );
  assert.throws(() =>
    validateOptimizationPilot({ ...options, latest: { ...options.latest, complete: false } }),
  );
});

test('a verified optimizer checkpoint cannot authenticate a different exported model', () => {
  const options = {
    latest: { step: 128, checkpoint_sha256: 'checkpoint' },
    index: {
      complete: true,
      step: 128,
      checkpoint_sha256: 'checkpoint',
      model_sha256: 'model',
      files: {
        'checkpoint.pt': 'checkpoint',
        'model.blzn': 'model',
        'verification.json': 'verification',
      },
    },
    verification: { passed: true },
    checkpointSha256: 'checkpoint',
    modelSha256: 'model',
    verificationSha256: 'verification',
  };
  assert.doesNotThrow(() => validateOptimizationCheckpoint(options));
  for (const patch of [
    { modelSha256: 'stale' },
    { checkpointSha256: 'corrupt' },
    { verificationSha256: 'changed' },
    { verification: { passed: false } },
    { latest: { ...options.latest, step: 256 } },
  ])
    assert.throws(
      () => validateOptimizationCheckpoint({ ...options, ...patch }),
      /published journal/,
    );
});
