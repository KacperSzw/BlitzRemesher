import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { runTeacherProfile } from '../scripts/neural/teacher-profile.mjs';
const hash = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value));
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-resident-teacher-'));
  const model = root + '/model.blzn';
  const count = 64 * 129 + 64 * 65 + 12 * 65;
  const weights = Buffer.alloc(24 + count * 4);
  weights.write('BLZNET02');
  weights.writeUInt32LE(4, 8);
  weights.writeUInt32LE(count, 12);
  weights.writeUInt32LE(64, 20);
  fs.writeFileSync(model, Buffer.concat([weights, Buffer.from(hash(weights))]));
  const binary = root + '/binary';
  fs.writeFileSync(binary, 'a frozen native binary');
  const source = root + '/source.mesh';
  fs.writeFileSync(source, 'verified geometry');
  const asset = {
    id: 'fixture',
    split: 'development',
    files: [{ path: source, sha256: hash(fs.readFileSync(source)) }],
  };
  const corpus = root + '/corpus.json';
  write(corpus, { assets: [asset] });
  const plan = {
    version: 1,
    architecture: 4,
    hidden_width: 64,
    profile: 'coverage',
    mask_only_coverage: true,
    teacher_selection: 'policy-mixed',
    pool: 4,
    source_limit: 3,
    adjacent_limit: 2,
    seed: 211,
    raster_backend: 'vulkan',
    vertex_storage: 'packed',
    candidate_batch: 4,
    workers: 2,
    gpu_memory_mib: 512,
    host_cache_mib: 128,
    measured_passes: 1,
    max_seconds: 30,
    job_seconds: 20,
    model,
    model_sha256: hash(fs.readFileSync(model)),
    corpus,
    training_selection: corpus,
    conditions: [
      {
        name: 'source',
        asset: 'fixture',
        pixels: 64,
        previous_steps: 0,
        previous_pixels: 0,
        preserve_uv: true,
        policy_rollout_trials: 0,
        simplifier: false,
        retained: 1,
        states: 2,
      },
      {
        name: 'previous',
        asset: 'fixture',
        pixels: 32,
        previous_steps: 1,
        previous_pixels: 64,
        preserve_uv: false,
        policy_rollout_trials: 64,
        simplifier: false,
        retained: 0.75,
        states: 3,
      },
    ],
  };
  const planPath = root + '/plan.json';
  write(planPath, plan);
  return { root, binary, model, source, plan, planPath };
}
function nativeArtifacts(binary, args, changed = '', version = 5) {
  assert.equal(args[0], '--jobs');
  const request = JSON.parse(fs.readFileSync(args[1])),
    output = args[2];
  if (request.teacher_strategy) version = 6;
  const coreFirst = request.teacher_strategy === 'coverage-core-first';
  fs.mkdirSync(output);
  const binaryHash = hash(fs.readFileSync(binary));
  const jobs = [],
    warmupCount = request.workers * 2;
  for (let id = 0; id < warmupCount + request.conditions.length; ++id) {
    const warmup = id < warmupCount,
      condition = warmup ? 0 : id - warmupCount;
    const c = request.conditions[condition];
    const directory = (warmup ? 'warmup-' : 'measured-') + id,
      folder = output + '/' + directory;
    fs.mkdirSync(folder);
    const contract = {
      teacher_version: changed === 'unsupported_version' ? 7 : version,
      ...(version === 6 && changed !== 'missing_strategy'
        ? { teacher_strategy: request.teacher_strategy ?? 'exhaustive' }
        : {}),
      schema: 4,
      pool: 4,
      training_profile: 'coverage',
      mask_only_coverage: true,
      candidate_batch: request.candidate_batch,
      seed: request.seed,
      states_requested: c.states,
      pixels: c.pixels,
      previous_steps: c.previous_steps,
      previous_pixels: c.previous_steps ? c.previous_pixels : c.pixels,
      source_limit: 3,
      adjacent_limit: 2,
      preserve_uv: c.preserve_uv,
      policy_action_candidates: true,
      policy_rollout_trials: c.policy_rollout_trials,
      gpu_memory_mib: request.gpu_memory_mib,
      raster: 'vulkan-v1',
      vertex_storage: 'packed',
      asset: { id: c.asset },
      binary_sha256: binaryHash,
    };
    if (changed === 'non_strategy_condition') contract.source_manifest_sha256 = 'changed';
    write(folder + '/contract.json', contract);
    const payload =
      'teacher labels ' + condition + (changed === 'labels' || coreFirst ? ' changed' : '');
    const episode =
      'audited geometry ' + condition + (changed === 'episode' || coreFirst ? ' changed' : '');
    fs.writeFileSync(folder + '/actions.bin', payload);
    fs.writeFileSync(folder + '/episode.bin', episode);
    const timings = Object.fromEntries(
      [
        'load_and_session',
        'packing_and_baseline',
        'state_setup',
        'policy_rollout',
        'predecessor_snapshot_and_audit',
        'features_and_proposals',
        'candidate_build',
        'candidate_audit',
        'exact_confirmation',
        'commit',
        'final_audit',
        'final_output',
        'other_preparation',
      ].map((k) => [k + '_seconds', 0.01]),
    );
    const audit = { complete: true, passed: true, error: changed === 'semantic' ? 0.2 : 0.1 };
    const index = {
      schema: 4,
      status: 'complete',
      complete: changed !== 'incomplete',
      training_started: false,
      reference_confirmed: true,
      requested_condition_available: changed !== 'unavailable',
      preceding_lod_emitted: c.previous_steps > 0,
      states: c.states + c.previous_steps,
      queries: coreFirst ? 30 : 50,
      source_triangles: 100,
      teacher_triangles: 96,
      previous_triangles: c.previous_steps ? 98 : 100,
      accepted: c.states + c.previous_steps,
      source_audit: audit,
      adjacent_audit: audit,
      baseline: audit,
      seed: { accepted: c.policy_rollout_trials > 0 },
      timing_version: 2,
      seconds: 0.12,
      total_wall_seconds: 0.13,
      timings,
      audit: { resource_failures: 0 },
      sha256: hash(Buffer.from(payload)),
      episode_sha256: hash(Buffer.from(episode)),
      contract_sha256: hash(fs.readFileSync(folder + '/contract.json')),
    };
    if (changed === 'timing') index.seconds = 0.2;
    if (changed === 'resource') index.audit.resource_failures = 1;
    if (changed === 'fewer_states') {
      --index.states;
      --index.accepted;
      index.status = 'search_exhausted';
    }
    write(folder + '/index.json', index);
    write(folder + '/trajectory.json', [
      {
        revision: c.previous_steps,
        queries: Array.from({ length: index.queries }, () => ({ known_mask: 27 })),
        candidate_search: [
          {
            from: 0,
            to: 1,
            queried_mask: coreFirst ? 1047 : 2047,
            ...(coreFirst ? { expanded: false, exact_rejected_mask: 0 } : {}),
          },
        ],
        ...(coreFirst
          ? {
              confirmations: [
                {
                  candidate: 0,
                  known: true,
                  passed: true,
                  source_passed: true,
                  adjacent_passed: true,
                },
              ],
            }
          : {}),
      },
    ]);
    write(folder + '/reuse.json', {
      duplicate_proposals: 1,
      identical_adjacent_audits: 2,
      pruned_candidates: 3,
      unbeatable_incumbent_skips: 4,
    });
    if (changed === 'missing_trace') fs.unlinkSync(folder + '/trajectory.json');
    if (changed === 'forged_hash') fs.appendFileSync(folder + '/actions.bin', 'tamper');
    jobs.push({
      id,
      condition,
      pass: 0,
      phase: warmup ? 'warmup' : 'measured',
      directory,
      complete: true,
      recovered: changed === 'recovered',
      worker: changed === 'cold_worker' ? 0 : id % request.workers,
      submitted_ns: id * 1e9,
      started_ns: id * 1e9 + 1,
      completed_ns: id * 1e9 + 5e8,
      consumed_ns: id * 1e9 + 5e8 + 1,
      index,
      payload_sha256: index.sha256,
      episode_sha256: index.episode_sha256,
    });
  }
  if (changed === 'order') jobs.reverse();
  if (changed === 'timestamps') jobs[1].started_ns = -1;
  write(output + '/report.json', {
    version: 1,
    complete: true,
    training_started: false,
    optimizer_updates: changed === 'training' ? 1 : 0,
    binary_sha256: binaryHash,
    request,
    request_sha256: hash(fs.readFileSync(args[1])),
    measured_seconds: 1,
    warmup_seconds: 1,
    total_wall_seconds: 3,
    jobs,
  });
}
test('resident ABBA compares actual payloads, episodes, conditions and warm worker timings', async () => {
  const f = fixture();
  let calls = 0,
    clock = 1000;
  try {
    const r = await runTeacherProfile({
      baseline: f.binary,
      optimized: f.binary,
      model: f.model,
      directory: f.root + '/out',
      plan: f.planPath,
      deadline: 200000,
      now: () => clock,
      execute: async (binary, args, options) => {
        ++calls;
        assert.ok(options.maximum > 0 && options.maximum <= 32000);
        nativeArtifacts(binary, args);
        clock += 4000;
        return { code: 0, signal: null, timed_out: false, cancelled: false, success: true };
      },
    });
    assert.equal(calls, 4);
    assert.equal(r.complete, true);
    assert.equal(r.matching_payloads, true);
    assert.equal(r.measured_speedup, 1);
    assert.equal(r.process_speedup, 1);
    assert.equal(r.training_started, false);
    assert.equal(r.optimizer_updates, 0);
    assert.deepEqual(
      r.rows.map((x) => x.variant),
      ['baseline', 'optimized', 'optimized', 'baseline'],
    );
  } finally {
    fs.rmSync(f.root, { recursive: true, force: true });
  }
});
for (const failure of [
  'labels',
  'episode',
  'semantic',
  'incomplete',
  'unavailable',
  'timing',
  'resource',
  'forged_hash',
  'recovered',
  'cold_worker',
  'order',
  'timestamps',
  'training',
  'signal',
  'timeout',
  'unsupported_version',
])
  test('resident profile suppresses every speedup for ' + failure, async () => {
    const f = fixture();
    let calls = 0;
    try {
      const r = await runTeacherProfile({
        baseline: f.binary,
        optimized: f.binary,
        model: f.model,
        directory: f.root + '/out',
        plan: f.planPath,
        deadline: Date.now() + 200000,
        execute: async (binary, args) => {
          nativeArtifacts(binary, args, ++calls === 2 ? failure : '');
          return calls === 2 && failure === 'signal'
            ? { code: null, signal: 'SIGTERM', success: false }
            : calls === 2 && failure === 'timeout'
              ? { code: 0, signal: null, success: false, timed_out: true }
              : { code: 0, signal: null, success: true, timed_out: false, cancelled: false };
        },
      });
      assert.equal(calls, 2);
      assert.equal(r.complete, false);
      assert.ok(r.error);
      assert.equal(r.speedup, null);
      assert.equal(r.measured_speedup, null);
      assert.equal(r.process_speedup, null);
      assert.equal(r.rows[0].complete, true);
      assert.equal(r.rows[1].complete, false);
    } finally {
      fs.rmSync(f.root, { recursive: true, force: true });
    }
  });
test('resident profile checks deadline, cancellation, source and model before execution', async () => {
  for (const failure of ['deadline', 'cancelled', 'source', 'model']) {
    const f = fixture();
    try {
      const controller = new AbortController();
      if (failure === 'cancelled') controller.abort();
      if (failure === 'source') fs.appendFileSync(f.source, 'changed');
      if (failure === 'model') fs.appendFileSync(f.model, 'changed');
      const r = await runTeacherProfile({
        baseline: f.binary,
        optimized: f.binary,
        model: f.model,
        directory: f.root + '/out',
        plan: f.planPath,
        deadline: failure === 'deadline' ? 0 : Date.now() + 200000,
        signal: controller.signal,
        execute: () => assert.fail('invalid inputs must not invoke a process'),
      });
      assert.equal(r.complete, false);
      assert.equal(r.speedup, null);
      assert.ok(r.error);
    } finally {
      fs.rmSync(f.root, { recursive: true, force: true });
    }
  }
});

test('resident parity permits only the documented exhaustive v5 to v6 contract transition', async () => {
  for (const failure of ['', 'missing_strategy']) {
    const f = fixture();
    let calls = 0;
    try {
      const r = await runTeacherProfile({
        baseline: f.binary,
        optimized: f.binary,
        model: f.model,
        directory: f.root + '/out',
        plan: f.planPath,
        deadline: Date.now() + 200000,
        execute: async (binary, args) => {
          const optimized = ++calls === 2 || calls === 3;
          nativeArtifacts(binary, args, optimized ? failure : '', optimized ? 6 : 5);
          return { code: 0, signal: null, success: true };
        },
      });
      assert.equal(r.complete, !failure);
      assert.equal(r.speedup, failure ? null : 1);
      assert.equal(calls, failure ? 2 : 4);
    } finally {
      fs.rmSync(f.root, { recursive: true, force: true });
    }
  }
});

test('resident strategy ABBA reports changed targets without claiming parity or quality', async () => {
  const f = fixture();
  let calls = 0;
  try {
    const r = await runTeacherProfile({
      baseline: f.binary,
      optimized: f.binary,
      model: f.model,
      comparison: 'strategy',
      timingAuthority: 'isolated_remote',
      directory: f.root + '/out',
      plan: f.planPath,
      deadline: Date.now() + 200000,
      execute: async (binary, args) => {
        ++calls;
        nativeArtifacts(binary, args);
        return { code: 0, signal: null, success: true };
      },
    });
    assert.equal(r.complete, true, r.error);
    assert.equal(calls, 4);
    assert.equal(r.strategy_measured_speedup, 1);
    assert.equal(r.strategy_fresh_states_speedup, 1);
    assert.equal(r.strategy_fresh_scored_queries_speedup, 0.6);
    assert.equal(r.measured_speedup, null);
    assert.equal(r.speedup, null);
    assert.equal(r.quality_proven, false);
    assert.equal(r.matching_payloads, undefined);
    assert.equal(r.matching_non_strategy_conditions, true);
    assert.equal(r.matching_repeats, true);
    assert.equal(r.timing_authority, 'isolated_remote');
    assert.equal(r.strategy_comparisons.length, 2);
    for (const pair of r.strategy_comparisons) {
      assert.equal(pair.labels_changed, true);
      assert.equal(pair.episode_changed, true);
      assert.equal(pair.baseline.queries, 50);
      assert.equal(pair.optimized.queries, 30);
      assert.equal(pair.optimized.expanded_edges, 0);
      assert.equal(pair.optimized.exact_confirmations, 1);
      assert.equal(pair.baseline.exact_confirmations, null);
    }
  } finally {
    fs.rmSync(f.root, { recursive: true, force: true });
  }
});

test('strategy throughput cannot improve merely by exhausting with fewer fresh states', async () => {
  const f = fixture();
  try {
    const r = await runTeacherProfile({
      baseline: f.binary,
      optimized: f.binary,
      model: f.model,
      comparison: 'strategy',
      directory: f.root + '/out',
      plan: f.planPath,
      deadline: Date.now() + 200000,
      execute: async (binary, args) => {
        const request = JSON.parse(fs.readFileSync(args[1]));
        nativeArtifacts(
          binary,
          args,
          request.teacher_strategy === 'coverage-core-first' ? 'fewer_states' : '',
        );
        return { code: 0, signal: null, success: true };
      },
    });
    assert.equal(r.complete, true, r.error);
    assert.equal(r.strategy_measured_speedup, 1);
    assert.equal(r.strategy_fresh_states_speedup, 0.6);
    assert.equal(r.equal_fresh_state_counts, false);
    assert.equal(r.strategy_throughput.baseline.fresh_states, 10);
    assert.equal(r.strategy_throughput.optimized.fresh_states, 6);
  } finally {
    fs.rmSync(f.root, { recursive: true, force: true });
  }
});

for (const failure of [
  'non_strategy_condition',
  'missing_trace',
  'repeat',
  'different_binary',
  'incomplete',
])
  test('resident strategy suppresses timings for ' + failure, async () => {
    const f = fixture();
    let calls = 0;
    try {
      const optimized = f.root + '/candidate';
      fs.writeFileSync(
        optimized,
        failure === 'different_binary' ? 'different' : fs.readFileSync(f.binary),
      );
      const r = await runTeacherProfile({
        baseline: f.binary,
        optimized,
        model: f.model,
        comparison: 'strategy',
        directory: f.root + '/out',
        plan: f.planPath,
        deadline: Date.now() + 200000,
        execute: async (binary, args) => {
          ++calls;
          nativeArtifacts(
            binary,
            args,
            failure === 'repeat' ? (calls === 3 ? 'semantic' : '') : calls === 2 ? failure : '',
          );
          return { code: 0, signal: null, success: true };
        },
      });
      assert.equal(r.complete, false);
      assert.equal(r.strategy_measured_speedup, null);
      assert.equal(r.strategy_process_speedup, null);
      assert.equal(r.speedup, null);
      assert.ok(r.error);
      assert.equal(calls, failure === 'different_binary' ? 0 : failure === 'repeat' ? 3 : 2);
    } finally {
      fs.rmSync(f.root, { recursive: true, force: true });
    }
  });
