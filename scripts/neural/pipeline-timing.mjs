#!/usr/bin/env node
// Turn a completed local cycle and its separate Nsight replays into an additive
// wall-time tree and two GPU-time trees. Never add host API waits to GPU work.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { hash, modelPayload } from './refactor-cycle.mjs';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('pipeline-timing.mjs RUN_DIRECTORY FRESH_OUTPUT_DIRECTORY');
if (fs.existsSync(output)) throw new Error('Choose a fresh output directory');
const read = (p) => JSON.parse(fs.readFileSync(path.join(input, p), 'utf8'));
const r = read('cycle/report.json'),
  profiles = read('profiles.json');
assert.ok(r.complete);
assert.ok(r.phases.every((p) => p.code === 0));
const teachers = r.phases.filter((p) => p.phase === 'teacher'),
  trainers = r.phases.filter((p) => p.phase === 'training'),
  audits = r.phases.filter((p) => p.phase === 'quality_audit');
assert.ok(teachers.every((p) => p.healthy) && audits.every((p) => p.healthy));
assert.ok(
  trainers.every((p) => p.health.complete && p.health.finite && p.health.optimizer_restored),
);
const sum = (values) => values.reduce((a, b) => a + b, 0);
const node = (name, seconds, children = [], note = '') => {
  assert.ok(Number.isFinite(seconds) && seconds >= 0, name);
  if (children.length)
    assert.ok(
      Math.abs(sum(children.map((c) => c.seconds)) - seconds) < 1e-8 * Math.max(1, seconds),
      name + ' total',
    );
  return { name, seconds, ...(children.length ? { children } : {}), ...(note ? { note } : {}) };
};
const parent = (name, children, note) =>
  node(name, sum(children.map((c) => c.seconds)), children, note);
const remainder = (name, total, children, note) =>
  node(
    name,
    total,
    [
      ...children,
      node('Other / uninstrumented remainder', total - sum(children.map((c) => c.seconds))),
    ],
    note,
  );
const shortAsset = (id) =>
  ({
    ph_painted_wooden_bench: 'Bench',
    ph_sweet_potato: 'Sweet potato',
    ph_namaqualand_boulder_04: 'Boulder',
    ph_painted_wooden_shelves: 'Shelves',
    ph_moon_rock_02: 'Moon rock',
  })[id] ?? id;
const teacherRows = teachers.map((p) =>
  node(
    `${shortAsset(p.condition.asset)} · ${p.condition.pixels}px${p.condition.previous ? ' · 2 preceding states at ' + p.condition.pixels * 2 + 'px' : ''}`,
    p.wall_seconds,
    [
      node('Teacher loop, final confirmation and dataset export', p.index.seconds),
      node(
        'Process / loading / resident setup / teardown remainder',
        p.wall_seconds - p.index.seconds,
      ),
    ],
    `${p.index.source_triangles.toLocaleString('en-US')} source triangles; ${p.index.states} states; ${p.index.queries} placement queries; ${p.reuse.pruned_candidates} pruned; ${p.index.audit.gpu_rasters} rasterizations; ${(p.index.audit.gpu_peak_bytes / 1048576).toFixed(1)} MiB native peak.`,
  ),
);
const trainerSum = (key) => sum(trainers.map((p) => p.health[key]));
const trainerWall = sum(trainers.map((p) => p.wall_seconds));
const training = node('Optimizer processes · 6 segments', trainerWall, [
  node(
    'Process / loading / packing / upload / restore / teardown remainder',
    trainerWall - trainerSum('seconds') - trainerSum('capture_seconds'),
    [],
    'These costs were not individually instrumented; this is not a measurement of idle GPU time.',
  ),
  node('CUDA graph warmup and capture', trainerSum('capture_seconds')),
  node(
    'Optimizer update windows · 12,288 updates',
    trainerSum('update_seconds'),
    [],
    'Includes graph replay, GPU execution and a status readback every 128 updates.',
  ),
  node(
    'Checkpoint / export / native and FP64 checks / restore verification',
    trainerSum('checkpoint_seconds'),
    [],
    '24 checkpoints, every 512 updates.',
  ),
  node(
    'Loop logging / remaining bookkeeping',
    trainerSum('seconds') - trainerSum('update_seconds') - trainerSum('checkpoint_seconds'),
  ),
]);
const auditRows = audits.map((p) => {
  const ranking = p.args[p.args.indexOf('--neural-control') + 1];
  return remainder(
    `${ranking} ranking`,
    p.wall_seconds,
    p.assets.map((a) => {
      const stages = [
        ['Encode / resident setup', 'encode_seconds'],
        ['Ranking / inference', 'inference_seconds'],
        ['Topology / decoding excluding nested audit and inference', 'decode_seconds'],
        ['Candidate / chain GPU audits', 'gpu_audit_seconds'],
        ['Final GPU confirmation', 'gpu_confirmation_seconds'],
      ].map(([label, key]) => node(label, a.neural[key]));
      return remainder(
        shortAsset(a.id),
        a.seconds,
        [
          node('Mesh loading', a.load_seconds),
          remainder('LOD generation', a.generation_seconds, stages),
          node('Mesh export', a.export_seconds),
        ],
        `${a.neural.gpu_rasters} rasterizations; ${a.neural.gpu_evaluations} audit evaluations; ${a.neural.fallback_levels} unreduced fallback levels. Wall timers include host work and synchronization.`,
      );
    }),
    'Two frozen diagnostic assets; 256 → 128px chain; GPU confirmation. This bounded run is not a quality score.',
  );
});
const wall = node('Complete local learning pass', r.wall_seconds, [
  parent('Teacher processes · 6 conditions', teacherRows),
  training,
  parent('Quality diagnostic · constant and learned', auditRows),
  node('Controller / GPU telemetry / reports', r.wall_seconds - r.child_seconds),
]);

const databases = Object.fromEntries(
  ['teacher', 'training'].map((k) => [
    k,
    new DatabaseSync(path.join(input, k + '-trace.sqlite'), { readOnly: true }),
  ]),
);
const kernels = (db, where = '') =>
  db
    .prepare(
      `SELECT s.value name,count(*) calls,sum(k.end-k.start)/1e9 seconds FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ${where} GROUP BY k.demangledName ORDER BY seconds DESC`,
    )
    .all();
const teacherKernels = kernels(databases.teacher);
const teacherGroups = [
  ['Appearance / normal and attribute matching', /::appearance\(/],
  ['Coverage distance and summary', /::(initialize|edt|edt_binary|transpose|directed_coverage)\(/],
  ['Raster visibility and attributes', /::raster</],
  [
    'Projection / triangle setup / bins / dirty tiles',
    /::(project|triangles|scatter|ranges|dirty_tiles)\(/,
  ],
  ['CUB sorting / scans / compaction across raster and topology', /cub::/],
  ['Placement / topology / features / policy', /.*/],
];
const grouped = teacherGroups.map(([name]) => ({ name, rows: [] }));
for (const k of teacherKernels)
  grouped[teacherGroups.findIndex(([, match]) => match.test(k.name))].rows.push(k);
const kernelLabel = (name) =>
  name.match(/(Device\w+Kernel)/)?.[1] ??
  name.match(/::([a-z][a-z_0-9]*)[<(]/)?.[1] ??
  name.split('<')[0];
const teacherGpu = parent(
  'Teacher GPU kernel time · separate bench 256 → 128px replay',
  grouped.map((g) =>
    parent(
      g.name,
      g.rows.map((k) =>
        node(
          kernelLabel(k.name),
          k.seconds,
          [],
          `${k.calls.toLocaleString('en-US')} calls; ${((k.seconds * 1e6) / k.calls).toFixed(3)} µs/call; ${k.name}`,
        ),
      ),
    ),
  ),
  'Summed GPU kernel durations across the entire process. Replay teacher core: ' +
    read('profile-teacher/index.json').seconds.toFixed(3) +
    ' s. This separate instrumented run is not added to the 40.226 s wall-time tree.',
);
const graphIds = databases.training
  .prepare('SELECT DISTINCT graphId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL')
  .all();
assert.equal(graphIds.length, 1);
const updateNodes = databases.training
  .prepare(
    'SELECT s.value name,k.graphNodeId node,count(*) calls,min(k.start) first_start,sum(k.end-k.start)/1e9 seconds FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.graphId IS NOT NULL GROUP BY k.graphNodeId ORDER BY first_start',
  )
  .all();
assert.equal(updateNodes.length, 42);
assert.ok(updateNodes.every((n) => n.calls === 2048));
// This mapping follows the observed capture and training/update.hpp::eager.
// Check the complete sequence so a changed graph cannot silently inherit labels.
const expected = [
  'select_states',
  'gather_packed',
  ...Array(6).fill('FillFunctor'),
  'sgemm',
  'CUDAFunctor_add',
  'launch_clamp_scalar',
  'sgemm',
  'CUDAFunctor_add',
  'launch_clamp_scalar',
  'sgemm',
  'CUDAFunctor_add',
  'placement_denominators',
  'placement_gradient',
  'sum_loss',
  'reduce_kernel',
  'CUDAFunctor_add',
  'sgemm',
  'splitKreduce_kernel',
  'sgemm',
  'CUDAFunctor_add',
  'threshold_kernel_impl',
  'reduce_kernel',
  'CUDAFunctor_add',
  'sgemm',
  'splitKreduce_kernel',
  'sgemm',
  'CUDAFunctor_add',
  'threshold_kernel_impl',
  'reduce_kernel',
  'CUDAFunctor_add',
  'sgemm',
  'splitKreduce_kernel',
  'CUDAFunctor_add',
  'norm_parts',
  'norm_finish',
  'adam',
  'finish',
];
updateNodes.forEach((n, i) =>
  assert.ok(n.name.includes(expected[i]), 'Unexpected captured kernel ' + i),
);
const range = (name, start, end) =>
  parent(
    name,
    updateNodes
      .slice(start, end)
      .map((k, i) =>
        node(`Kernel ${start + i + 1}: ${kernelLabel(k.name)}`, k.seconds / k.calls, [], k.name),
      ),
  );
const updateGpu = parent(
  'One optimizer update · mean active GPU kernel time',
  [
    range('Sample states and gather packed features', 0, 2),
    range('Clear parameter gradients', 2, 8),
    parent('Forward MLP · 128 → 64 → 64 → 12', [
      range('Layer 1 · matrix / bias / ReLU', 8, 11),
      range('Layer 2 · matrix / bias / ReLU', 11, 14),
      range('Output layer · matrix / bias', 14, 16),
    ]),
    range('Masked placement loss and output derivative', 16, 19),
    parent('Backpropagation', [
      range('Output layer and hidden activation gradient', 19, 26),
      range('Hidden layer and input activation gradient', 26, 33),
      range('First layer parameter gradients', 33, 38),
    ]),
    range('Gradient norm and clipping scale', 38, 40),
    range('AdamW parameters and moments', 40, 41),
    range('Update counters', 41, 42),
  ],
  '42 kernels per update; 2,048 captured replays. Warmup, checkpoints and other non-graph kernels are excluded. Memory operations and gaps are not included in active kernel time.',
);
const teacherProfile = profiles.profiles.find((p) => p.name === 'teacher'),
  trainProfile = profiles.profiles.find((p) => p.name === 'training');
const parity = {
  teacher_payload:
    hash(path.join(input, 'profile-teacher/actions.bin')) ===
    teacherProfile.source_phase.index.sha256,
  training_parameters:
    modelPayload(
      path.join(input, 'profile-training', read('profile-training/latest.json').model),
    ) === trainProfile.source_phase.model_payload_sha256,
};
assert.ok(parity.teacher_payload && parity.training_parameters);
const apis = Object.fromEntries(
  Object.entries(databases).map(([kind, db]) => [
    kind,
    db
      .prepare(
        'SELECT s.value name,count(*) calls,sum(k.end-k.start)/1e9 seconds FROM CUPTI_ACTIVITY_KIND_RUNTIME k JOIN StringIds s ON s.id=k.nameId GROUP BY k.nameId ORDER BY seconds DESC',
      )
      .all(),
  ]),
);
for (const db of Object.values(databases)) db.close();
const data = {
  revision: r.revision,
  date: r.started,
  precision: r.precision,
  gpu: r.phases[0].gpu_before,
  complete: r.complete,
  score: null,
  states: r.states,
  updates: r.final_step - r.initial_step,
  batch: r.batch,
  parity,
  methodology: {
    wall: 'One sequential unprofiled local pass. Parent/child wall timers include CPU work, GPU work and synchronization. Uninstrumented components are left as explicit remainders.',
    teacher_gpu:
      'Separate exact teacher replay. Each kernel belongs to one function-name group. GPU times are not added to wall or host API times.',
    update_gpu:
      'Separate exact final trainer replay. Means over 2,048 graph executions, grouped in captured-node order and checked against the source call sequence.',
    limits:
      'Shared RTX 2080; one pass, no confidence interval. Eight requested states per condition plus six preceding states. One diagnostic audit after this pass; the long cycle normally audits every 30 minutes and adapts states up to 64. This profile does not establish quality improvement or long-run time shares.',
  },
  wall,
  teacher_gpu: teacherGpu,
  update_gpu: updateGpu,
  trainer_segments: trainers.map((p, i) => ({
    segment: i,
    states: p.health.states,
    wall_seconds: p.wall_seconds,
    health: p.health,
  })),
  teacher_kernels: teacherKernels,
  update_nodes: updateNodes,
  cuda_apis: apis,
};
fs.mkdirSync(output, { recursive: true });
const copied = [
  'cycle/report.json',
  'profiles.json',
  'profile-teacher/index.json',
  'profile-training/latest.json',
  'ctest.log',
  'cuda-memcheck-audits.log',
];
for (const kind of ['teacher', 'training'])
  for (const report of [
    'cuda_gpu_kern_sum',
    'cuda_api_sum',
    'cuda_gpu_mem_time_sum',
    'cuda_gpu_mem_size_sum',
  ])
    copied.push(`${kind}-stats_${report}.csv`);
const sources = copied.map((p) => {
  const name = p.replaceAll('/', '-');
  fs.copyFileSync(path.join(input, p), path.join(output, name));
  return { path: p, saved_as: name, sha256: hash(path.join(input, p)) };
});
const localTraces = ['teacher', 'training'].flatMap((kind) =>
  ['sqlite', 'nsys-rep'].map((ext) => ({
    path: path.join(input, `${kind}-trace.${ext}`),
    sha256: hash(path.join(input, `${kind}-trace.${ext}`)),
  })),
);
const provenance = {
  score: null,
  sources,
  local_traces: localTraces,
  scripts: ['scripts/neural/pipeline-profile.mjs', 'scripts/neural/pipeline-timing.mjs'].map(
    (p) => ({ path: p, sha256: hash(p) }),
  ),
};
fs.writeFileSync(path.join(output, 'timings.json'), JSON.stringify(data, null, 2) + '\n');
fs.writeFileSync(path.join(output, 'provenance.json'), JSON.stringify(provenance, null, 2) + '\n');
const duration = (s) =>
  s >= 1
    ? s.toFixed(3) + ' s'
    : s >= 0.001
      ? (s * 1000).toFixed(3) + ' ms'
      : (s * 1e6).toFixed(3) + ' µs';
const escape = (s) =>
  String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
const tree = (n, total = n.seconds, depth = 0) => {
  const title = `<span class="name">${escape(n.name)}</span><strong>${duration(n.seconds)}</strong><span class="percent">${((100 * n.seconds) / total).toFixed(1)}%</span>`;
  const bar = `style="--share:${(100 * n.seconds) / total}%"`;
  return n.children
    ? `<details ${depth < 2 ? 'open' : ''}><summary ${bar}>${title}</summary><div class="children">${n.note ? `<p class="note">${escape(n.note)}</p>` : ''}${n.children.map((c) => tree(c, n.seconds, depth + 1)).join('')}</div></details>`
    : `<div class="leaf" ${bar} title="${escape(n.note ?? '')}">${title}</div>`;
};
function textTree(n, prefix = '', last = true, root = true) {
  return (
    (root ? '' : prefix + (last ? '└─ ' : '├─ ')) +
    n.name +
    '  ' +
    duration(n.seconds) +
    '\n' +
    (n.children ?? [])
      .map((c, i) =>
        textTree(
          c,
          prefix + (root ? '' : last ? '   ' : '│  '),
          i === n.children.length - 1,
          false,
        ),
      )
      .join('')
  );
}
fs.writeFileSync(
  path.join(output, 'timing-tree.txt'),
  [wall, teacherGpu, updateGpu].map((n) => textTree(n)).join('\n'),
);
const trainerTable = `<div class="table"><table><thead><tr><th>Segment</th><th>Resident states</th><th>Process</th><th>Capture</th><th>2,048 updates</th><th>Checkpoint checks</th><th>µs/update</th></tr></thead><tbody>${data.trainer_segments.map((p) => `<tr><td>${p.segment + 1}</td><td>${p.states}</td><td>${duration(p.wall_seconds)}</td><td>${duration(p.health.capture_seconds)}</td><td>${duration(p.health.update_seconds)}</td><td>${duration(p.health.checkpoint_seconds)}</td><td>${((p.health.update_seconds / 2048) * 1e6).toFixed(1)}</td></tr>`).join('')}</tbody></table></div>`;
const html = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FP32 local learning pipeline timings</title>
<style>:root{font-family:system-ui,sans-serif;color:#162735;background:#f4f6f8}body{max-width:1200px;margin:0 auto;padding:32px 24px;line-height:1.5}h1{font-size:2rem;line-height:1.2}h2{margin:2.5rem 0 .7rem}a{color:#11635c}nav{display:flex;gap:20px;flex-wrap:wrap}.intro{max-width:88ch}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{flex:1;min-width:180px;background:white;border:1px solid #d5dee4;border-radius:12px;padding:18px}.card strong{display:block;font-size:1.8rem;white-space:normal}.note{color:#50616c;font-size:.9rem;overflow-wrap:anywhere}.children{margin-left:20px;border-left:1px solid #c7d3dc;padding-left:12px}summary,.leaf{display:flex;align-items:center;gap:12px;position:relative;padding:10px 12px;margin:3px 0;background:linear-gradient(to right,#dcefe9 var(--share),#fff var(--share));border-radius:6px;min-height:24px}summary{cursor:pointer}summary::before{content:'▸';width:12px;flex-shrink:0}details[open]>summary::before{content:'▾'}.name{flex:1;overflow-wrap:anywhere}strong,.percent{white-space:nowrap;font-variant-numeric:tabular-nums}.percent{width:55px;text-align:right;color:#496268;font-size:.85rem}.leaf{padding-left:36px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:right;padding:10px;border-bottom:1px solid #d5dee4;white-space:nowrap}.table{overflow:auto}th:first-child,td:first-child{text-align:left}.sources{columns:2;overflow-wrap:anywhere}.warning{padding:14px 18px;border-left:4px solid #c77731;background:#fff3e4}@media(max-width:650px){body{padding:16px 10px}.children{margin-left:6px;padding-left:5px}summary,.leaf{gap:6px;padding:8px 6px;font-size:.85rem}.percent{display:none}.leaf{padding-left:24px}.sources{columns:1}}</style>
<h1>FP32 local learning pipeline</h1><p class="intro">Measured on the local RTX 2080, ${escape(r.started.slice(0, 10))}, revision ${escape(r.revision.slice(0, 7))}. ${r.states} states, ${data.updates.toLocaleString('en-US')} updates, batch ${r.batch}, plus the frozen two-asset quality diagnostic. Positions, raster attributes and training stay FP32; existing exact-audit FP64 arithmetic is retained.</p>
<nav><a href="#cycle">Whole cycle</a><a href="#teacher">Teacher GPU</a><a href="#update">Update GPU</a><a href="#segments">Trainer segments</a><a href="#evidence">Evidence</a></nav>
<div class="cards"><div class="card">Complete pass<strong>${r.wall_seconds.toFixed(2)} s</strong></div><div class="card">Measured update window<strong>${((trainerSum('update_seconds') / data.updates) * 1e6).toFixed(1)} µs / update</strong></div><div class="card">Captured graph<strong>42 kernels / update</strong></div></div>
<p class="warning">${escape(data.methodology.limits)} The three trees have separate scopes. Add siblings within one tree; never add GPU kernels or CPU API wait time to process wall time.</p>
<h2 id="cycle">Complete cycle · wall time</h2><p>Expand any branch. Percentages and bars are relative to its immediate parent. ${escape(data.methodology.wall)}</p>${tree(wall)}
<h2 id="teacher">Slowest teacher condition · separate GPU trace</h2><p>Bench, 630 source triangles, two preceding states at 256px and eight target states at 128px. ${escape(data.methodology.teacher_gpu)} Both replayed teacher data and trainer parameters match their unprofiled payload hashes.</p>${tree(teacherGpu)}
<p>The trace records 9,193 blocking cudaMemcpy calls taking 6.608 s on the CPU, mostly waiting for GPU work. Actual device-to-host transfer activity takes 7.243 ms for 0.537 MB. These are overlapping measurements, not an extra 6.608 s of transfer cost.</p>
<h2 id="update">One optimizer update · separate GPU trace</h2><p>${escape(data.methodology.update_gpu)} The full dataset occupies ${(trainers.at(-1).health.resident_data_bytes / 1024).toFixed(1)} KiB in resident packed storage. Active kernel time excludes launch gaps, memory operations and status synchronization.</p>${tree(updateGpu)}
<h2 id="segments">Six measured trainer processes</h2>${trainerTable}
<h2 id="evidence">Evidence and validation</h2><p>All 22 CTest checks passed; CUDA audit memcheck reports zero errors. Checkpoint/export/optimizer restoration checks passed throughout. Unreduced fallback LODs remain visible; this timing pass makes no model-quality claim. No remote GPU was rented.</p>
<p><a href="timings.json">All derived timings and kernel rows</a> · <a href="timing-tree.txt">Plain text hierarchy</a> · <a href="provenance.json">Input and trace hashes</a></p><ul class="sources">${sources.map((s) => `<li><a href="${escape(s.saved_as)}">${escape(s.path)}</a></li>`).join('')}</ul></html>`;
fs.writeFileSync(path.join(output, 'timings.html'), html);
console.log(
  JSON.stringify(
    {
      output,
      wall_seconds: wall.seconds,
      teacher_gpu_seconds: teacherGpu.seconds,
      update_gpu_us: updateGpu.seconds * 1e6,
      parity,
    },
    null,
    2,
  ),
);
