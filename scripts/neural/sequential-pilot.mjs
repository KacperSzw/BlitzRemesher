// A separate small-source learning gate; the full-size pilot remains unchanged.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { read, write } from './artifacts.mjs';
import { boundedProcess } from './bounded-process.mjs';
import { comparePolicyRanking } from './policy-ranking-quality.mjs';

const digest = (p) => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const members = (assets) =>
  assets.map(({ id, category }) => ({ id, category })).sort((a, b) => a.id.localeCompare(b.id));

export function evaluateSequentialPilot(gate, manifest, audits) {
  if (
    gate.version !== 1 ||
    !gate.seeds.length ||
    new Set(gate.seeds).size !== gate.seeds.length ||
    !isDeepStrictEqual(gate.uv_modes, [true, false]) ||
    !(
      gate.minimum_relative_retained_improvement > 0 &&
      gate.minimum_relative_retained_improvement < 1
    ) ||
    !(gate.maximum_relative_per_lod_increase >= 0 && gate.maximum_relative_per_lod_increase < 1)
  )
    throw Error('Invalid sequential pilot gate');
  const expected = members(manifest.assets),
    seen = new Set(),
    modelHashes = new Map(),
    comparisons = [];
  if (!expected.length || new Set(expected.map((x) => x.id)).size !== expected.length)
    throw Error('Invalid sequential pilot assets');
  for (const { seed, preserve_uv, before, after } of audits) {
    const key = `${seed}:${preserve_uv}`;
    if (!gate.seeds.includes(seed) || !gate.uv_modes.includes(preserve_uv) || seen.has(key))
      throw Error('Duplicate or unexpected sequential pilot comparison');
    seen.add(key);
    for (const [model, audit] of [
      ['initial', before],
      [seed, after],
    ]) {
      if (!/^[a-f0-9]{64}$/.test(audit.model_sha256))
        throw Error('Missing sequential pilot model identity');
      if (modelHashes.has(model) && modelHashes.get(model) !== audit.model_sha256)
        throw Error('Sequential pilot model changed between comparisons');
      modelHashes.set(model, audit.model_sha256);
      const execution = audit.execution_settings;
      if (
        !isDeepStrictEqual(members(audit.expected_assets), expected) ||
        execution.preserve_uv !== preserve_uv ||
        execution.action_batch !== 1 ||
        !execution.source_preparation ||
        execution.source_preparation.algorithm !== 'quadric-coupled-rebuild-merged-v1'
      )
        throw Error('Sequential pilot inputs, UV mode or execution scope differ');
      const limit = execution.source_preparation.maximum_triangles;
      if (!Number.isInteger(limit) || limit < 4) throw Error('Invalid prepared source limit');
      for (const row of audit.rows)
        if (
          !row.prepared_source ||
          !/^[a-f0-9]{64}$/.test(row.prepared_source.sha256) ||
          row.prepared_source.triangles !== row.lods[0].triangles ||
          row.lods[0].triangles > limit
        )
          throw Error('Missing or oversized prepared pilot source');
    }
    const comparison = comparePolicyRanking(before, after);
    comparisons.push({
      seed,
      preserve_uv,
      ...comparison,
      passed:
        comparison.relative_retained_improvement >= gate.minimum_relative_retained_improvement &&
        comparison.worst_relative_lod_increase <= gate.maximum_relative_per_lod_increase,
    });
  }
  if (seen.size !== gate.seeds.length * gate.uv_modes.length)
    throw Error('Incomplete sequential pilot comparison matrix');
  return {
    complete: true,
    passed: comparisons.every((x) => x.passed),
    score: null,
    scope: 'small-source learning feasibility; full-size scale and shading remain unverified',
    comparisons,
  };
}

export async function runSequentialPilot({
  gatePath,
  initial,
  models,
  directory,
  signal,
  gpuMemoryMiB = 1024,
  maximumMinutes = 180,
}) {
  if (
    !Number.isFinite(maximumMinutes) ||
    maximumMinutes <= 0 ||
    maximumMinutes > 240 ||
    !Number.isInteger(gpuMemoryMiB) ||
    gpuMemoryMiB < 128 ||
    gpuMemoryMiB > 65536
  )
    throw Error('Invalid sequential pilot resource bounds');
  const gate = read(gatePath),
    manifest = read(gate.manifest);
  if (
    digest(gate.manifest) !== gate.manifest_sha256 ||
    digest(gate.settings) !== gate.settings_sha256
  )
    throw Error('Frozen sequential pilot inputs changed');
  if (
    !isDeepStrictEqual(
      models.map((m) => m.seed).sort((a, b) => a - b),
      [...gate.seeds].sort((a, b) => a - b),
    )
  )
    throw Error('Provide every frozen learner seed exactly once');
  if (fs.existsSync(directory)) throw Error('Choose a fresh sequential pilot directory');
  const binary = 'build/neural/blitz-neural-diagnostics',
    binaryHash = digest(binary);
  const entries = [
    { name: 'initial', path: initial },
    ...models.map((m) => ({ ...m, name: 'seed-' + m.seed })),
  ].map((m) => ({ ...m, sha256: digest(m.path) }));
  const report = {
    complete: false,
    passed: false,
    score: null,
    gate,
    gate_sha256: digest(gatePath),
    revision: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    binary_sha256: binaryHash,
    models: entries,
    started_at: new Date().toISOString(),
    timing_authority: 'local_shared',
    tasks: [],
    audits: [],
  };
  const deadline = Date.now() + maximumMinutes * 60000;
  const persist = () => write(path.join(directory, 'report.json'), report);
  persist();
  const env = { ...process.env };
  env.VK_DRIVER_FILES ??= '/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json';
  for (const key of ['DISPLAY', 'WAYLAND_DISPLAY', 'VK_ICD_FILENAMES']) delete env[key];
  try {
    for (const preserve_uv of gate.uv_modes) {
      const settings = { ...read(gate.settings), preserve_uv, gpu_memory_mib: gpuMemoryMiB };
      const settingsPath = path.join(directory, `settings-uv${Number(preserve_uv)}.json`);
      write(settingsPath, settings);
      for (const model of entries) {
        if (digest(binary) !== binaryHash || digest(model.path) !== model.sha256)
          throw Error('Pilot binary or model changed');
        const name = `uv${Number(preserve_uv)}-${model.name}`,
          output = path.join(directory, name + '.json');
        const fd = fs.openSync(path.join(directory, name + '.log'), 'wx'),
          began = Date.now();
        let result;
        try {
          result = await boundedProcess(
            binary,
            ['--audit-model', gate.manifest, model.path, settingsPath, output],
            {
              maximum: Math.min(50 * 60000, deadline - began),
              signal,
              env,
              stdio: ['ignore', fd, fd],
            },
          );
        } finally {
          fs.closeSync(fd);
        }
        report.tasks.push({ name, seconds: (Date.now() - began) / 1000, process: result });
        persist();
        console.log(JSON.stringify(report.tasks.at(-1)));
        const audit = read(output);
        if (!result.success || !audit.complete) throw Error('Incomplete pilot audit: ' + name);
        if (
          audit.model_sha256 !== model.sha256 ||
          audit.binary_sha256 !== binaryHash ||
          audit.manifest_sha256 !== gate.manifest_sha256
        )
          throw Error('Pilot audit identity differs from pinned inputs: ' + name);
        report.audits.push({
          seed: model.seed,
          model: model.name,
          preserve_uv,
          output,
          sha256: digest(output),
        });
        persist();
      }
    }
    const pairs = report.audits
      .filter((a) => a.model !== 'initial')
      .map((a) => ({
        seed: a.seed,
        preserve_uv: a.preserve_uv,
        after: read(a.output),
        before: read(
          report.audits.find((b) => b.preserve_uv === a.preserve_uv && b.model === 'initial')
            .output,
        ),
      }));
    Object.assign(report, evaluateSequentialPilot(gate, manifest, pairs));
  } catch (error) {
    report.error = String(error);
    throw error;
  } finally {
    report.finished_at = new Date().toISOString();
    persist();
  }
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 6)
    throw Error('sequential-pilot.mjs GATE INITIAL_MODEL MODELS_JSON FRESH_DIRECTORY');
  const cancellation = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => cancellation.abort());
  await runSequentialPilot({
    gatePath: process.argv[2],
    initial: process.argv[3],
    models: read(process.argv[4]),
    directory: process.argv[5],
    signal: cancellation.signal,
  });
}
