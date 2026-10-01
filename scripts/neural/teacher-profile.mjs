#!/usr/bin/env node
// Resident teacher-only comparisons. Timing claims require complete, matching
// artifacts; neither this runner nor the native --jobs path starts an optimizer.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { boundedProcess } from './bounded-process.mjs';
import { read, write } from './artifacts.mjs';
import { readinessModel } from './teacher-readiness.mjs';
const hash = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const order = ['baseline', 'optimized', 'optimized', 'baseline'];
const finite = (value) => Number.isFinite(value) && value >= 0;
const stageFields = [
  'load_and_session_seconds',
  'packing_and_baseline_seconds',
  'state_setup_seconds',
  'policy_rollout_seconds',
  'predecessor_snapshot_and_audit_seconds',
  'features_and_proposals_seconds',
  'candidate_build_seconds',
  'candidate_audit_seconds',
  'exact_confirmation_seconds',
  'commit_seconds',
  'final_audit_seconds',
  'final_output_seconds',
  'other_preparation_seconds',
];
const semanticFields = [
  'schema',
  'status',
  'complete',
  'source_triangles',
  'teacher_triangles',
  'previous_triangles',
  'states',
  'accepted',
  'reference_confirmed',
  'requested_condition_available',
  'preceding_lod_emitted',
  'seed',
  'source_audit',
  'adjacent_audit',
  'baseline',
];

export function validateTeacherRun(directory, request, binarySha256) {
  const report = read(path.join(directory, 'report.json'));
  const measuredCount = request.conditions.length * request.measured_passes;
  const warmupCount = request.workers * 2;
  if (
    report.version !== 1 ||
    report.complete !== true ||
    report.training_started !== false ||
    report.optimizer_updates !== 0 ||
    report.binary_sha256 !== binarySha256 ||
    !isDeepStrictEqual(report.request, request) ||
    !finite(report.measured_seconds) ||
    report.measured_seconds <= 0 ||
    !finite(report.warmup_seconds) ||
    !finite(report.total_wall_seconds) ||
    !Array.isArray(report.jobs) ||
    report.jobs.length !== warmupCount + measuredCount
  )
    throw new Error('incomplete or mismatched resident teacher report');
  const outputs = [];
  let previousConsumed = 0;
  const warmed = new Set();
  for (const [position, row] of report.jobs.entries()) {
    const warmup = position < warmupCount;
    const condition = warmup ? 0 : (position - warmupCount) % request.conditions.length;
    const pass = warmup ? 0 : Math.floor((position - warmupCount) / request.conditions.length);
    if (
      row.id !== position ||
      row.condition !== condition ||
      row.pass !== pass ||
      row.phase !== (warmup ? 'warmup' : 'measured') ||
      row.complete !== true ||
      row.recovered !== false ||
      !Number.isInteger(row.worker) ||
      row.worker < 0 ||
      row.worker >= request.workers ||
      typeof row.directory !== 'string' ||
      !/^(warmup|measured)-[0-9]+$/.test(row.directory)
    )
      throw new Error('invalid resident job order, worker, or recovery state');
    const times = ['submitted_ns', 'started_ns', 'completed_ns', 'consumed_ns'].map((k) => row[k]);
    if (
      times.some((n) => !Number.isSafeInteger(n) || n < 0) ||
      times.some((n, i) => i && n < times[i - 1]) ||
      times[3] < previousConsumed
    )
      throw new Error('invalid worker lifecycle timestamps');
    previousConsumed = times[3];
    if (warmup) warmed.add(row.worker);
    const folder = path.join(directory, row.directory);
    const index = read(path.join(folder, 'index.json'));
    const contract = read(path.join(folder, 'contract.json'));
    if (
      !isDeepStrictEqual(index, row.index) ||
      index.complete !== true ||
      index.schema !== 4 ||
      index.training_started !== false ||
      index.reference_confirmed !== true ||
      index.requested_condition_available !== true ||
      index.states <= 0 ||
      index.audit?.resource_failures !== 0 ||
      index.timing_version !== 2 ||
      !stageFields.every((k) => finite(index.timings?.[k]))
    )
      throw new Error('incomplete, unavailable, resource-limited or untimed teacher job');
    const phaseSum = stageFields
      .filter((k) => k !== 'load_and_session_seconds')
      .reduce((n, k) => n + index.timings[k], 0);
    if (
      !finite(index.seconds) ||
      !finite(index.total_wall_seconds) ||
      Math.abs(phaseSum - index.seconds) > 1e-8 * Math.max(1, index.seconds) ||
      Math.abs(index.timings.load_and_session_seconds + index.seconds - index.total_wall_seconds) >
        1e-8 * Math.max(1, index.total_wall_seconds)
    )
      throw new Error('teacher phase timings are not additive');
    for (const [file, key, rowKey] of [
      ['actions.bin', 'sha256', 'payload_sha256'],
      ['episode.bin', 'episode_sha256', 'episode_sha256'],
    ]) {
      const actual = hash(path.join(folder, file));
      if (actual !== index[key] || actual !== row[rowKey])
        throw new Error('teacher payload or episode checksum mismatch');
    }
    if (
      hash(path.join(folder, 'contract.json')) !== index.contract_sha256 ||
      contract.binary_sha256 !== binarySha256
    )
      throw new Error('teacher contract checksum or binary mismatch');
    const c = request.conditions[condition];
    const expected = {
      schema: 4,
      pool: 4,
      training_profile: 'coverage',
      mask_only_coverage: true,
      candidate_batch: request.candidate_batch,
      seed: request.seed,
      states_requested: c.states,
      pixels: c.pixels,
      previous_steps: c.previous_steps,
      previous_pixels: c.previous_steps
        ? c.previous_pixels || Math.min(512, c.pixels * 2)
        : c.pixels,
      source_limit: 3,
      adjacent_limit: 2,
      preserve_uv: c.preserve_uv,
      policy_action_candidates: true,
      policy_rollout_trials: c.policy_rollout_trials,
      gpu_memory_mib: request.gpu_memory_mib,
      raster: 'vulkan',
      vertex_storage: 'packed',
    };
    for (const [key, value] of Object.entries(expected))
      if (!isDeepStrictEqual(contract[key], value))
        throw new Error('teacher request mismatch: ' + key);
    if (
      contract.asset?.id !== c.asset ||
      (c.previous_steps > 0 && !index.preceding_lod_emitted) ||
      ((c.simplifier || c.policy_rollout_trials) && index.seed?.accepted !== true)
    )
      throw new Error('requested teacher condition was not exercised');
    const normalizedContract = { ...contract };
    delete normalizedContract.binary_sha256;
    outputs.push({
      phase: row.phase,
      condition,
      pass,
      payload_sha256: row.payload_sha256,
      episode_sha256: row.episode_sha256,
      contract: normalizedContract,
      semantics: Object.fromEntries(semanticFields.map((k) => [k, index[k]])),
    });
  }
  if (warmed.size !== request.workers) throw new Error('not every worker was warmed');
  return { report, outputs };
}

export async function runTeacherProfile({
  baseline,
  optimized,
  directory,
  model,
  plan = 'research/neural/teacher-profile.json',
  deadline,
  signal,
  execute = boundedProcess,
  workers,
  candidateBatch,
  gpuMemoryMiB,
  maxSeconds,
  now = Date.now,
  timingAuthority = 'local_shared',
}) {
  if (fs.existsSync(directory)) throw new Error('choose a fresh teacher profile directory');
  fs.mkdirSync(directory, { recursive: true });
  const report = {
    version: 1,
    mode: 'parity',
    complete: false,
    training_started: false,
    optimizer_updates: 0,
    quality_proven: false,
    score: null,
    speedup: null,
    measured_speedup: null,
    process_speedup: null,
    order,
    rows: [],
    timing_authority: timingAuthority,
  };
  const save = () => write(path.join(directory, 'report.json'), report);
  try {
    if (!Number.isFinite(deadline) || deadline <= now() || signal?.aborted)
      throw new Error('teacher profile deadline/cancellation');
    const request = read(plan);
    const checked = readinessModel(model ?? request.model);
    if (checked.sha256 !== request.model_sha256 || checked.width !== request.hidden_width)
      throw new Error('frozen teacher policy changed');
    request.model = path.resolve(model ?? request.model);
    if (workers !== undefined) request.workers = workers;
    if (candidateBatch !== undefined) request.candidate_batch = candidateBatch;
    if (gpuMemoryMiB !== undefined) request.gpu_memory_mib = gpuMemoryMiB;
    if (maxSeconds !== undefined) request.max_seconds = maxSeconds;
    if (
      ![1, 2].includes(request.workers) ||
      ![1, 2, 4, 8].includes(request.candidate_batch) ||
      !Number.isInteger(request.measured_passes) ||
      request.measured_passes < 1 ||
      request.measured_passes > 8 ||
      !Array.isArray(request.conditions) ||
      !request.conditions.length ||
      request.conditions.length > 64 ||
      !Number.isInteger(request.gpu_memory_mib) ||
      request.gpu_memory_mib < 128 ||
      request.gpu_memory_mib > 65536 ||
      !Number.isInteger(request.max_seconds) ||
      request.max_seconds < 1 ||
      request.max_seconds > 1800
    )
      throw new Error('invalid resident teacher profile settings');
    request.max_seconds = Math.min(request.max_seconds, Math.floor((deadline - now()) / 4000) - 4);
    if (!Number.isInteger(request.max_seconds) || request.max_seconds < 1)
      throw new Error('insufficient teacher profile deadline');
    const binaries = { baseline: path.resolve(baseline), optimized: path.resolve(optimized) };
    report.binary_sha256 = Object.fromEntries(
      Object.entries(binaries).map(([k, p]) => [k, hash(p)]),
    );
    report.plan_sha256 = hash(plan);
    report.model = checked;
    report.request = request;
    report.manifests = {};
    const corpus = read(request.corpus),
      allowed = new Set(read(request.training_selection).assets.map((a) => a.id));
    const files = new Map();
    for (const c of request.conditions) {
      const asset = corpus.assets.find((a) => a.id === c.asset);
      if (!asset || asset.split !== 'development' || !allowed.has(c.asset))
        throw new Error('benchmark condition outside training development selection');
      for (const file of asset.files) {
        if (files.has(file.path) && files.get(file.path) !== file.sha256)
          throw new Error('conflicting benchmark source checksums');
        files.set(file.path, file.sha256);
      }
    }
    for (const [file, expected] of files)
      if (hash(file) !== expected) throw new Error('benchmark source checksum mismatch: ' + file);
    for (const file of [
      request.corpus,
      request.training_selection,
      fileURLToPath(new URL('../../research/PROTOCOL.md', import.meta.url)),
    ])
      report.manifests[file] = hash(file);
    const requestPath = path.resolve(directory, 'request.json');
    write(requestPath, request);
    report.request_sha256 = hash(requestPath);
    save();
    let reference;
    for (const [repeat, variant] of order.entries()) {
      if (signal?.aborted || deadline <= now())
        throw new Error('teacher profile deadline/cancellation');
      const output = path.resolve(directory, `${repeat}-${variant}`);
      const row = { repeat, variant, directory: output, complete: false };
      report.rows.push(row);
      const log = fs.openSync(output + '.log', 'wx');
      const start = now();
      try {
        row.process = await execute(binaries[variant], ['--jobs', requestPath, output], {
          maximum: Math.min((request.max_seconds + 2) * 1000, deadline - now() - 1000),
          grace: 1000,
          signal,
          stdio: ['ignore', log, log],
        });
        row.wall_seconds = (now() - start) / 1000;
        if (
          !row.process.success ||
          row.process.code !== 0 ||
          row.process.signal ||
          row.process.timed_out ||
          row.process.cancelled
        )
          throw new Error('resident teacher process failed: ' + JSON.stringify(row.process));
        const verified = validateTeacherRun(output, request, report.binary_sha256[variant]);
        if (verified.report.request_sha256 !== report.request_sha256)
          throw new Error('resident request checksum mismatch');
        row.result = verified.report;
        if (reference && !isDeepStrictEqual(verified.outputs, reference))
          throw new Error('teacher payload, episode or semantic outcomes differ');
        reference ??= verified.outputs;
        row.complete = true;
      } catch (error) {
        row.error = String(error);
        throw error;
      } finally {
        fs.closeSync(log);
        save();
      }
    }
    if (signal?.aborted || deadline <= now())
      throw new Error('teacher profile deadline/cancellation');
    const sum = (variant, key) =>
      report.rows
        .filter((r) => r.variant === variant)
        .reduce((total, row) => total + (key === 'wall_seconds' ? row[key] : row.result[key]), 0);
    report.complete = true;
    report.matching_payloads = true;
    report.matching_semantics = true;
    report.measured_speedup =
      sum('baseline', 'measured_seconds') / sum('optimized', 'measured_seconds');
    report.speedup = report.measured_speedup;
    const optimizedWall = sum('optimized', 'wall_seconds');
    report.process_speedup =
      optimizedWall > 0 ? sum('baseline', 'wall_seconds') / optimizedWall : null;
  } catch (error) {
    report.error = String(error);
  } finally {
    report.finished = now();
    save();
  }
  return report;
}

async function legacyProfile(args) {
  const [baseline, optimized, directory, model, only] = args;
  if (!baseline || !optimized || !directory)
    throw new Error('teacher-profile.mjs BASELINE OPTIMIZED FRESH_OUTPUT [MODEL] [CASE]');
  if (fs.existsSync(directory)) throw new Error('Choose a fresh profile directory');
  fs.mkdirSync(directory, { recursive: true });
  const hash = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const cases = [
    { name: 'bench32', asset: 'ph_painted_wooden_bench', pixels: 32, previous: 0, states: 8 },
    { name: 'organic64', asset: 'ph_sweet_potato', pixels: 64, previous: 0, states: 8 },
    {
      name: 'bench-adjacent64',
      asset: 'ph_painted_wooden_bench',
      pixels: 64,
      previous: 2,
      states: 6,
    },
    { name: 'organic-adjacent64', asset: 'ph_sweet_potato', pixels: 64, previous: 2, states: 6 },
    ...(model
      ? [
          {
            name: 'policy-adjacent64',
            asset: 'ph_sweet_potato',
            pixels: 64,
            previous: 2,
            states: 6,
            model,
          },
          {
            name: 'policy-adjacent32',
            asset: 'ph_sweet_potato',
            pixels: 32,
            previous: 2,
            states: 6,
            model,
          },
        ]
      : []),
  ].filter((c) => !only || c.name === only);
  if (!cases.length) throw new Error('Unknown comparison case');
  const report = {
    baseline_sha256: hash(baseline),
    optimized_sha256: hash(optimized),
    model_sha256: model ? hash(model) : null,
    order: ['baseline', 'optimized', 'optimized', 'baseline'],
    score: null,
    cases,
    rows: [],
  };
  const save = () =>
    fs.writeFileSync(path.join(directory, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  save();
  const gpu = () =>
    execFileSync(
      'nvidia-smi',
      ['--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
      { encoding: 'utf8', timeout: 2000 },
    ).trim();
  for (const c of cases)
    for (const [repeat, variant] of report.order.entries()) {
      const output = path.resolve(directory, `${c.name}-${repeat}-${variant}`),
        binary = path.resolve(variant === 'baseline' ? baseline : optimized);
      const args = [
        c.asset,
        output,
        '--states',
        String(c.states),
        '--pool',
        '4',
        '--pixels',
        String(c.pixels),
        '--previous-steps',
        String(c.previous),
        '--minutes',
        '3',
        '--gpu-memory-mib',
        '256',
        '--seed',
        '101',
        ...(c.model ? ['--model', path.resolve(c.model)] : []),
      ];
      const before = gpu(),
        start = performance.now(),
        log = fs.openSync(output + '.log', 'w');
      const process = await boundedProcess(binary, args, {
        maximum: 185000,
        grace: 2000,
        stdio: ['ignore', log, log],
      });
      const code = process.success ? 0 : process.code || -1;
      fs.closeSync(log);
      const wall = (performance.now() - start) / 1000,
        index = fs.existsSync(output + '/index.json')
          ? JSON.parse(fs.readFileSync(output + '/index.json', 'utf8'))
          : null;
      const row = {
        case: c.name,
        variant,
        repeat,
        code,
        wall_seconds: wall,
        gpu_before: before,
        gpu_after: gpu(),
        args,
        index,
      };
      if (fs.existsSync(output + '/reuse.json'))
        row.reuse = JSON.parse(fs.readFileSync(output + '/reuse.json', 'utf8'));
      report.rows.push(row);
      save();
      console.log(
        JSON.stringify({
          case: c.name,
          variant,
          repeat,
          code,
          wall_seconds: wall,
          teacher_seconds: index?.seconds,
          states: index?.states,
          sha256: index?.sha256,
        }),
      );
    }
  report.comparisons = cases.map((c) => {
    const rows = report.rows.filter((r) => r.case === c.name),
      complete = rows.every((r) => r.code === 0 && r.index?.complete),
      hashes = [...new Set(rows.map((r) => r.index?.sha256))];
    const median = (a) => {
      a.sort((x, y) => x - y);
      return (a[Math.floor((a.length - 1) / 2)] + a[Math.floor(a.length / 2)]) / 2;
    };
    const metric = (variant, key) =>
      median(
        rows
          .filter((r) => r.variant === variant)
          .map((r) => (key === 'seconds' ? r.index.seconds : r.wall_seconds)),
      );
    return {
      case: c.name,
      complete,
      identical_training_payloads: complete && hashes.length === 1,
      hashes,
      ...(complete && hashes.length === 1
        ? {
            teacher_speedup: metric('baseline', 'seconds') / metric('optimized', 'seconds'),
            wall_speedup: metric('baseline', 'wall') / metric('optimized', 'wall'),
          }
        : {}),
    };
  });
  save();
  console.log(JSON.stringify(report.comparisons));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args[0] === '--jobs') {
    const [, plan, baseline, optimized, directory, model, workers] = args;
    if (!plan || !baseline || !optimized || !directory || args.length > 7)
      throw new Error(
        'teacher-profile.mjs --jobs PLAN BASELINE OPTIMIZED FRESH_OUTPUT [MODEL] [WORKERS]',
      );
    const controller = new AbortController();
    for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => controller.abort());
    const report = await runTeacherProfile({
      baseline,
      optimized,
      directory,
      model,
      plan,
      workers: workers === undefined ? undefined : Number(workers),
      deadline: Date.now() + 900000,
      signal: controller.signal,
    });
    process.exitCode = report.complete ? 0 : 1;
  } else await legacyProfile(args);
}
