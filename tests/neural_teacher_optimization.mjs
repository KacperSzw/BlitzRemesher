import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { runTeacherOptimization } from '../scripts/neural/teacher-optimization-job.mjs';
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
  return {
    directory: directory + '/results',
    model,
    request: {
      ...request(),
      initialization_sha256: createHash('sha256').update(fs.readFileSync(model)).digest('hex'),
    },
    deadline: 112 * 60000,
    now: () => 0,
  };
}

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
