#!/usr/bin/env node
// Bounded teacher preparation only; this never invokes an optimizer.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { boundedProcess } from './bounded-process.mjs';
import { read, write } from './artifacts.mjs';

export const readinessManifest = 'research/neural/action-diagnostic.json';
export const readinessFixtures = Object.freeze([
  { asset: 'ph_painted_wooden_shelves', previous_steps: 0 },
  { asset: 'ph_moon_rock_02', previous_steps: 1 },
]);
const batches = [1, 4, 4, 1];
const timingFields = [
  'load_and_session_seconds',
  'packing_and_baseline_seconds',
  'state_setup_seconds',
  'features_and_proposals_seconds',
  'candidate_build_seconds',
  'candidate_audit_seconds',
  'commit_seconds',
  'other_preparation_seconds',
];
const digest = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const fileDigest = (file) => digest(fs.readFileSync(file));

export function readinessModel(file) {
  const bytes = fs.readFileSync(file),
    magic = bytes.toString('ascii', 0, 8);
  if (bytes.length < 84 || !['BLZNET01', 'BLZNET02'].includes(magic))
    throw new Error('readiness requires a checksummed v4 placement model');
  const architecture = bytes.readUInt32LE(8),
    width = magic === 'BLZNET02' ? bytes.readUInt32LE(20) : 64;
  const header = magic === 'BLZNET02' ? 24 : 20,
    provenance = bytes.readUInt32LE(16);
  const count = width * 129 + width * (width + 1) + 12 * (width + 1);
  if (
    architecture !== 4 ||
    ![64, 128, 256].includes(width) ||
    provenance > 65536 ||
    bytes.readUInt32LE(12) !== count ||
    bytes.length !== header + provenance + count * 4 + 64 ||
    digest(bytes.subarray(0, -64)) !== bytes.toString('ascii', bytes.length - 64)
  )
    throw new Error('invalid readiness v4 model payload or checksum');
  for (let offset = header + provenance; offset < bytes.length - 64; offset += 4)
    if (!Number.isFinite(bytes.readFloatLE(offset))) throw new Error('nonfinite readiness model');
  return { architecture, width, sha256: digest(bytes) };
}

export function readinessAssets(root = process.cwd()) {
  const manifest = read(path.join(root, readinessManifest));
  if (manifest.score_eligible !== false || manifest.assets?.length !== readinessFixtures.length)
    throw new Error('readiness requires the fixed two-asset diagnostic manifest');
  return readinessFixtures.map(({ asset }) => {
    const entry = manifest.assets.find((entry) => entry.id === asset);
    if (!entry || entry.split !== 'development' || !entry.files?.length)
      throw new Error('missing development readiness fixture: ' + asset);
    return entry;
  });
}

export function readinessArguments(fixture, batch, model, output, maximum = 30000) {
  return [
    fixture.asset,
    output,
    '--architecture',
    '4',
    '--teacher-selection',
    'policy-mixed',
    '--corpus',
    readinessManifest,
    '--training-selection',
    readinessManifest,
    '--states',
    '2',
    '--pool',
    '4',
    '--pixels',
    '64',
    '--seed',
    '101',
    '--previous-steps',
    String(fixture.previous_steps),
    ...(fixture.previous_steps ? ['--previous-pixels', '128'] : []),
    '--source-limit',
    '3',
    '--adjacent-limit',
    '2',
    '--preserve-uv',
    'on',
    '--training-profile',
    'coverage',
    '--mask-only-coverage',
    'on',
    '--raster-backend',
    'vulkan',
    '--vertex-storage',
    'packed',
    '--audit-mode',
    'sparse',
    '--candidate-batch',
    String(batch),
    '--gpu-memory-mib',
    '512',
    '--model',
    model,
    '--minutes',
    String(maximum / 60000),
  ];
}

function payloadRows(file, states) {
  const bytes = fs.readFileSync(file);
  if (bytes.toString('ascii', 0, 8) !== 'BLZACT05' || bytes.readUInt32LE(8) !== 4)
    throw new Error('readiness requires compact v4 teacher rows');
  let offset = 12,
    rows;
  for (const [i, width] of [2, 1, 4, 1, 4].entries()) {
    if (offset + 8 > bytes.length) throw new Error('truncated teacher rows');
    const count = Number(bytes.readBigUInt64LE(offset));
    offset += 8;
    if (!Number.isSafeInteger(count) || count < 1 || count > (bytes.length - offset) / width)
      throw new Error('invalid teacher row count');
    if (i === 3) rows = count;
    if (
      i === 4 &&
      (count !== states + 1 ||
        bytes.readUInt32LE(offset) !== 0 ||
        bytes.readUInt32LE(offset + (count - 1) * width) !== rows)
    )
      throw new Error('teacher state offsets do not match completed states');
    offset += count * width;
  }
  return rows;
}

async function gpuObservation(directory, name, deadline, signal, execute, now) {
  const log = path.join(directory, name + '.log'),
    fd = fs.openSync(log, 'w');
  try {
    if (deadline <= now() || signal?.aborted)
      return { available: false, reason: 'deadline/cancellation' };
    const result = await execute(
      'nvidia-smi',
      [
        '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory',
        '--format=csv,noheader,nounits',
      ],
      { maximum: Math.min(2000, deadline - now()), grace: 0, signal, stdio: ['ignore', fd, fd] },
    );
    return {
      at: now(),
      available: result.success === true,
      result,
      csv: fs.readFileSync(log, 'utf8'),
    };
  } catch (error) {
    return { at: now(), available: false, error: String(error) };
  } finally {
    fs.closeSync(fd);
  }
}

export async function runTeacherReadiness({
  model,
  directory,
  deadline = Date.now() + 250000,
  signal,
  execute = boundedProcess,
  now = Date.now,
  timingAuthority = 'local_shared',
  expectedRequest,
}) {
  const report = {
    experiment: 'teacher-readiness',
    complete: false,
    training_started: false,
    score: null,
    quality_proven: false,
    speedup: null,
    deadline,
    timing_authority: timingAuthority,
    batch_order: batches,
    runs: [],
    comparisons: [],
  };
  if (fs.existsSync(directory)) throw new Error('use a fresh teacher readiness directory');
  fs.mkdirSync(directory, { recursive: true });
  try {
    if (!Number.isFinite(deadline) || deadline <= now() || signal?.aborted)
      throw new Error('readiness deadline/cancellation');
    report.model = readinessModel(model);
    readinessAssets();
    report.manifest_sha256 = fileDigest(readinessManifest);
    if (
      expectedRequest &&
      (report.model.sha256 !== expectedRequest.model?.sha256 ||
        report.manifest_sha256 !== expectedRequest.manifest_sha256)
    )
      throw new Error('immutable teacher readiness request checksum mismatch');
    report.gpu_before = await gpuObservation(
      directory,
      'gpu-before',
      deadline,
      signal,
      execute,
      now,
    );
    for (const fixture of readinessFixtures) {
      for (const [repeat, batch] of batches.entries()) {
        if (deadline <= now() || signal?.aborted)
          throw new Error('readiness deadline/cancellation');
        const output = path.join(directory, fixture.asset + '-' + repeat + '-batch' + batch);
        const maximum = Math.min(30000, deadline - now());
        const args = readinessArguments(fixture, batch, model, output, maximum);
        const run = {
          ...fixture,
          repeat,
          batch,
          command: 'build/neural/blitz-neural-placement-prepare',
          args,
          complete: false,
          rows: null,
          started: now(),
        };
        report.runs.push(run);
        const fd = fs.openSync(output + '.log', 'w');
        try {
          run.process = await execute(run.command, args, {
            maximum,
            grace: 0,
            signal,
            stdio: ['ignore', fd, fd],
          });
          run.wall_seconds = (now() - run.started) / 1000;
          if (fs.existsSync(output + '/index.json')) run.index = read(output + '/index.json');
          const index = run.index;
          run.timings = index?.timings;
          if (fs.existsSync(output + '/actions.bin'))
            run.payload_sha256 = fileDigest(output + '/actions.bin');
          if (fs.existsSync(output + '/episode.bin'))
            run.episode_sha256 = fileDigest(output + '/episode.bin');
          if (
            !run.process.success ||
            run.process.code !== 0 ||
            run.process.signal ||
            run.process.timed_out ||
            run.process.cancelled
          )
            throw new Error('teacher process failed or exceeded its deadline');
          if (
            !index ||
            index.schema !== 4 ||
            index.complete !== true ||
            index.status !== 'complete' ||
            index.training_started !== false ||
            index.reference_confirmed !== true ||
            index.requested_condition_available !== true ||
            index.preceding_lod_emitted !== Boolean(fixture.previous_steps) ||
            index.states !== 2 + fixture.previous_steps
          )
            throw new Error('teacher did not complete the requested source/predecessor states');
          if (
            !run.payload_sha256 ||
            !run.episode_sha256 ||
            run.payload_sha256 !== index.sha256 ||
            run.episode_sha256 !== index.episode_sha256 ||
            fileDigest(output + '/contract.json') !== index.contract_sha256
          )
            throw new Error('teacher artifact checksum mismatch');
          run.rows = payloadRows(output + '/actions.bin', index.states);
          if (
            !run.timings ||
            !timingFields.every(
              (name) => Number.isFinite(run.timings[name]) && run.timings[name] >= 0,
            )
          )
            throw new Error('missing or invalid teacher stage timings');
          if (deadline <= now() || signal?.aborted)
            throw new Error('readiness deadline/cancellation');
          run.complete = true;
        } catch (error) {
          run.error = String(error);
        } finally {
          fs.closeSync(fd);
          write(directory + '/report.json', report);
        }
      }
    }
    report.gpu_after = await gpuObservation(directory, 'gpu-after', deadline, signal, execute, now);
    for (const fixture of readinessFixtures) {
      const runs = report.runs.filter((run) => run.asset === fixture.asset);
      const complete = runs.length === 4 && runs.every((run) => run.complete);
      const same =
        complete &&
        new Set(runs.map((run) => run.payload_sha256 + ':' + run.episode_sha256)).size === 1;
      report.comparisons.push({ ...fixture, complete, matching_payloads: same, speedup: null });
    }
    report.complete =
      report.comparisons.every((entry) => entry.complete && entry.matching_payloads) &&
      deadline > now() &&
      !signal?.aborted;
    if (report.complete) {
      let a = 0,
        b = 0;
      for (const comparison of report.comparisons) {
        const runs = report.runs.filter((run) => run.asset === comparison.asset);
        const total = (batch) =>
          runs.filter((run) => run.batch === batch).reduce((sum, run) => sum + run.wall_seconds, 0);
        const first = total(1),
          second = total(4);
        comparison.speedup = second > 0 ? first / second : null;
        a += first;
        b += second;
      }
      report.speedup = b > 0 ? a / b : null;
    } else report.error = 'readiness incomplete or teacher payloads differ; no timing comparison';
  } catch (error) {
    report.error = String(error);
  } finally {
    report.finished = now();
    write(directory + '/report.json', report);
  }
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [model, directory] = process.argv.slice(2);
  if (!model || !directory || process.argv.length !== 4)
    throw new Error('teacher-readiness.mjs MODEL OUTPUT_DIRECTORY');
  const controller = new AbortController();
  for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => controller.abort());
  const report = await runTeacherReadiness({
    model: path.resolve(model),
    directory: path.resolve(directory),
    signal: controller.signal,
  });
  console.log(
    JSON.stringify({ complete: report.complete, report: path.resolve(directory, 'report.json') }),
  );
  process.exitCode = report.complete ? 0 : 1;
}
