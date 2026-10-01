// Matched controls for the coverage objective. Shading-objective savings are
// measured separately; they are never credited to the mask renderer.
import fs from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { read, write } from './artifacts.mjs';
import { hash, modelPayload } from './refactor-cycle.mjs';
export const initialModelSha = '9152bb42cb807a2e91fe3217ab6dc3bbcf11be12618bcd706acf71a8e3fff185';
export const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
export function pilotReady(report) {
  if (!report.complete || report.conditions.length !== 48) return false;
  const categories = new Set();
  for (const row of report.conditions) {
    const r = row.result;
    if (
      !row.complete ||
      !r?.complete ||
      !r.reference_confirmed ||
      !['complete', 'search_exhausted', 'predecessor_unavailable'].includes(r.status)
    )
      return false;
    if (r.states > 0) categories.add(r.category);
  }
  return ['manufactured', 'organic', 'rocks', 'stress'].every((c) => categories.has(c));
}
export async function benchmarkCoverage(
  { root, execute, phase },
  { model, memory = 16384, repeats = 5, quality = false, includeAttributes = false } = {},
) {
  fs.mkdirSync(root, { recursive: true });
  const curriculum = root + '/benchmark-curriculum.json';
  write(curriculum, {
    version: 1,
    conditions: [{ asset: 'ph_namaqualand_boulder_04', pixels: 128, previous_steps: 0 }],
  });
  const variants = [
      { name: 'legacy', mask: 'off', backend: 'reference' },
      { name: 'mask', mask: 'on', backend: 'reference' },
      { name: 'fused', mask: 'on', backend: 'fused' },
    ],
    rows = [];
  if (includeAttributes)
    variants.push({ name: 'attributes', mask: 'off', backend: 'reference', profile: 'attributes' });
  phase('matched-coverage-benchmark');
  for (let repeat = 0; repeat < repeats; ++repeat)
    for (const v of repeat % 2 ? [...variants].reverse() : variants) {
      const directory = root + `/matched-${v.name}-${repeat}`;
      const code = await execute(
        'blitz-neural-cycle',
        [
          directory,
          '--states',
          '2',
          '--updates',
          '1024',
          '--batch',
          '512',
          '--seed',
          '101',
          '--minutes',
          '3',
          '--gpu-memory-mib',
          String(memory),
          '--training-profile',
          v.profile ?? 'coverage',
          '--mask-only-coverage',
          v.mask,
          '--candidate-batch',
          '2',
          '--update-backend',
          v.backend,
          '--curriculum',
          curriculum,
          '--initialize',
          model,
          '--quality',
          quality ? 'on' : 'off',
        ],
        directory + '.log',
        3.1,
      );
      const report = read(directory + '/report.json');
      if (code !== 0 || !report.complete || report.step !== 1024)
        throw new Error('Incomplete matched coverage cycle');
      const shard = directory + '/data/' + report.all_datasets[0],
        index = read(shard + '/index.json');
      const row = {
        variant: v.name,
        repeat,
        seconds: report.seconds,
        phases: report.phases.map((p) => ({
          phase: p.phase,
          seconds: p.seconds,
          update_seconds: p.update_seconds,
          capture_seconds: p.capture_seconds,
          checkpoint_seconds: p.checkpoint_seconds,
          append_seconds: p.append_seconds,
        })),
        dataset_sha256: hash(shard + '/actions.bin'),
        model_payload: modelPayload(directory + '/' + report.checkpoint + '/model.blzn'),
        audit: index.audit,
        teacher_timings: index.timings,
        states: index.states,
      };
      rows.push(row);
      write(root + '/matched-coverage.json', { complete: false, rows });
    }
  const reference = rows.find((r) => r.variant === 'legacy');
  if (
    rows.some((r) => r.variant !== 'attributes' && r.dataset_sha256 !== reference.dataset_sha256) ||
    new Set(
      rows
        .filter((r) => r.variant !== 'fused' && r.variant !== 'attributes')
        .map((r) => r.model_payload),
    ).size !== 1
  )
    throw new Error('Mask renderer changed teacher data or reference optimizer output');
  const timings = Object.fromEntries(
    variants.map((v) => [
      v.name,
      median(rows.filter((r) => r.variant === v.name).map((r) => r.seconds)),
    ]),
  );
  // --check verifies fused gradients, optimizer moments and exact same-backend
  // restart. A whole-cycle win is required before changing the run backend.
  const selected_backend = timings.fused <= timings.mask * 0.95 ? 'fused' : 'reference';
  const result = {
    complete: true,
    rows,
    median_cycle_seconds: timings,
    selected_backend,
    mask_cycle_speedup: timings.legacy / timings.mask,
  };
  write(root + '/matched-coverage.json', result);
  return result;
}
export async function validateCoverage(ctx) {
  const { root, execute, phase } = ctx,
    model = '/workspace/initial-model.blzn';
  if (hash(model) !== initialModelSha) throw new Error('Warm-start model changed');
  phase('coverage-contracts');
  // Setup already ran CTest. Repeat the critical memory and restart contracts
  // under the exact allocated driver/device before generating fresh examples.
  await execute(
    'compute-sanitizer',
    [
      '--tool',
      'memcheck',
      '--error-exitcode',
      '1',
      'build/neural/blitz-neural-vulkan-tests',
      '--memcheck',
    ],
    root + '/mask-memcheck.log',
    3,
    { CUDA_MODULE_LOADING: 'EAGER', CUDA_MODULE_DATA_LOADING: 'EAGER' },
  );
  await execute('blitz-neural-vulkan-tests', [], root + '/vulkan-validation.log', 3, {
    VK_INSTANCE_LAYERS: 'VK_LAYER_KHRONOS_validation',
  });
  if (/Validation Error:/.test(fs.readFileSync(root + '/vulkan-validation.log', 'utf8')))
    throw new Error('Vulkan validation layer reported errors');
  await execute('blitz-neural-diagnostics', ['--check'], root + '/resident-contracts.log', 3);
  // The old strict replay must still reproduce its failure; repair is a new,
  // explicitly bounded representation, never a mutation of historical evidence.
  fs.writeFileSync(
    root + '/moon-original.json',
    gunzipSync(
      fs.readFileSync('research/neural/evidence/prepared-local/moon-packing-failure.json.gz'),
    ),
  );
  await execute(
    'blitz',
    ['audit-replay', root + '/moon-original.json', '--gpu-memory-mib', '16384'],
    root + '/moon-replay.json',
    2,
  );
  if (!read(root + '/moon-replay.json').reproduced)
    throw new Error('Historical packed failure no longer reproduces');
  await execute(
    'blitz-neural-packing-diagnostic',
    [
      '--repair',
      'data/polyhaven/moon_rock_02/source.gltf',
      'research/neural/refactor-smoke.json',
      root + '/moon-repaired',
      '32',
    ],
    root + '/moon-repair.log',
    2,
  );
  const repaired = read(root + '/moon-repaired/report.json');
  if (
    ['repaired', 'device', 'snapshot', 'snapshot_search'].some(
      (k) => !repaired[k]?.passed || !repaired[k]?.complete,
    )
  )
    throw new Error('Sparse exact position repair failed');
  await execute(
    'blitz-neural-action-train',
    ['--replay', 'research/neural/evidence/gpu-refactor-local/replay', root + '/cross-gpu.json'],
    root + '/cross-gpu.log',
    1,
  );
  if (!read(root + '/cross-gpu.json').passed) throw new Error('Cross-GPU export parity failed');
  const benchmark = await benchmarkCoverage(ctx, { model });
  phase('coverage-curriculum-validation');
  const code = await execute(
    'blitz-neural-pilot-prepare',
    [root + '/packed-pilot', '16384', '--training-profile', 'coverage'],
    root + '/packed-pilot.log',
    10.1,
  );
  const pilot = read(root + '/packed-pilot/report.json');
  if (code !== 0 || !pilotReady(pilot)) throw new Error('Coverage curriculum is not ready');
  await execute(
    'blitz-neural-hardware-profile',
    ['ph_namaqualand_boulder_04', root + '/raster.json', '128'],
    root + '/raster.log',
    1.5,
  );
  const raster = read(root + '/raster.json');
  if (!raster.complete || raster.rows.some((r) => r.coverage_only && r.coverage_changes))
    throw new Error('Mask raster coverage parity failed');
  const result = {
    complete: true,
    model,
    model_sha256: hash(model),
    selected_backend: benchmark.selected_backend,
    benchmark,
    pilot,
    raster,
    checkpoint_kind: 'shape_pretraining',
    release_quality_proven: false,
  };
  write(root + '/validation.json', result);
  return result;
}
export function coverageArchitecture(v) {
  const t = v.benchmark.median_cycle_seconds;
  const stages = (name) =>
    v.benchmark.rows.filter((r) => r.variant === name).flatMap((r) => r.phases);
  const stage = (variant, name) =>
    median(
      stages(variant)
        .filter((p) => p.phase === name)
        .map((p) => p.seconds),
    );
  const teacher = (name) =>
    median(
      v.benchmark.rows.filter((r) => r.variant === 'mask').map((r) => r.teacher_timings[name]),
    );
  const update = (name) =>
    median(
      stages('mask')
        .filter((p) => p.phase === 'training')
        .map((p) => p[name]),
    );
  const draws = (mask) =>
    v.raster.rows.filter((r) => r.storage === 'packed' && r.coverage_only === mask);
  const draw = (mask, key) => median(draws(mask).map((r) => r[key]));
  return `# Coverage pretraining before the two-hour run

\`\`\`mermaid
flowchart TD
  A["Packed mesh cache + immutable FP32 source"] --> B["GPU topology and candidate placements"]
  B --> C["One conservative Vulkan R8 pass: ${(draw(true, 'raster_gpu_seconds') * 1e6).toFixed(1)} µs/view"]
  C --> D["CUDA coverage witnesses; candidate audits: ${(teacher('candidate_audit_seconds') * 1000).toFixed(1)} ms/cycle"]
  D --> E["Source + predecessor hard gates"]
  E --> F["Compact examples; position supervision only"]
  F --> G["Captured 128 → 64 → 64 → 12 MLP + AdamW: ${(update('update_seconds') * 1000).toFixed(1)} ms / 1024 updates (reference)"]
  G --> B
  G --> H["Verified checkpoint + bounded recovery journal"]
  G --> I["Scheduled coverage gates + nonblocking shading diagnostics"]
\`\`\`

Five alternating runs with the same coverage objective, teacher inputs and update budget:

| Full cycle | Median seconds |
|---|---:|
| Legacy attachments, reference optimizer | ${t.legacy.toFixed(6)} |
| Mask-only, reference optimizer | ${t.mask.toFixed(6)} |
| Mask-only, fused optimizer | ${t.fused.toFixed(6)} |

Mask renderer whole-cycle speedup: ${(t.legacy / t.mask).toFixed(2)}×.
Median teacher time: ${stage('mask', 'teacher').toFixed(6)} s; training phase:
${stage('mask', 'training').toFixed(6)} s. Setup and final publication are included
in whole-cycle time. Per-phase, transfer, allocation and draw counters are in
matched-coverage.json; GPU packing/render/unpack timestamps are in raster.json.
Selected update backend: ${v.selected_backend}. Selection requires at least 5%
lower median full-cycle time plus numerical and restart checks.

Median stage timings for mask-only coverage with the reference optimizer:

| Stage | Milliseconds |
|---|---:|
| Teacher total | ${(stage('mask', 'teacher') * 1000).toFixed(3)} |
| ↳ Load and session | ${(teacher('load_and_session_seconds') * 1000).toFixed(3)} |
| ↳ Packing and baseline gates | ${(teacher('packing_and_baseline_seconds') * 1000).toFixed(3)} |
| ↳ GPU state setup | ${(teacher('state_setup_seconds') * 1000).toFixed(3)} |
| ↳ Features and proposals | ${(teacher('features_and_proposals_seconds') * 1000).toFixed(3)} |
| ↳ Candidate construction | ${(teacher('candidate_build_seconds') * 1000).toFixed(3)} |
| ↳ Candidate audits | ${(teacher('candidate_audit_seconds') * 1000).toFixed(3)} |
| ↳ Commit | ${(teacher('commit_seconds') * 1000).toFixed(3)} |
| ↳ Final gates, serialization and remaining preparation | ${(teacher('other_preparation_seconds') * 1000).toFixed(3)} |
| Training total | ${(stage('mask', 'training') * 1000).toFixed(3)} |
| ↳ Dataset append | ${(update('append_seconds') * 1000).toFixed(3)} |
| ↳ Graph capture | ${(update('capture_seconds') * 1000).toFixed(3)} |
| ↳ Optimizer updates | ${(update('update_seconds') * 1000).toFixed(3)} |
| ↳ Checkpoint snapshot/enqueue/wait | ${(update('checkpoint_seconds') * 1000).toFixed(3)} |

Medians of sub-stages need not add to the median total. Asynchronous checkpoint
publication overlaps the cycle. These are bounded cold cycles with two teacher
states; persistent training amortizes setup and graph capture.

Hardware profile: median packed full/mask GPU raster time
${(draw(false, 'raster_gpu_seconds') * 1e6).toFixed(1)} / ${(draw(true, 'raster_gpu_seconds') * 1e6).toFixed(1)} µs per view;
renderer workspace ${draw(false, 'gpu_bytes')} / ${draw(true, 'gpu_bytes')} bytes.
All mask/full coverage pixels match. Packing/interop wall time is reported
separately in raster.json; GPU draw speedup is not whole-cycle speedup.

Positions are UNORM16 in mesh bounds with at most 5% exact FP32 exceptions among
referenced surviving vertex IDs. Normals/tangents use 10-bit components and sign;
UVs use UNORM16 in [-8,8], colors RGBA8. Coverage draws skip attribute packing.
Reference masks use one bit/pixel; candidates can be read directly from R8 surfaces.
Remaining repeated work is teacher topology, candidate/view draws, exact fallback
distance transforms, and host decisions/synchronization. The masked objective
does not supervise appearance; these checkpoints require shading-aware fine-tuning.

Learning lasts at least 120 minutes, with 10 additional minutes for finalization.
Interrupted downtime is excluded. This development pilot is not a release score.
`;
}
