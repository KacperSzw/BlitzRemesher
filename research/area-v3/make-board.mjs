import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

// Builds a self-contained board from the frozen benchmark rows and exported
// geometry. It does not rerun reduction or substitute presentation views for
// the audit cameras.
const root = process.argv[2] ? resolve(process.argv[2]) : resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const output = join(root, 'research/area-v3/board');
const read = async path => JSON.parse(await readFile(path, 'utf8'));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const runSpecs = [
  {id: 'area-v3-uncapped', label: 'Uncapped reference', short: 'Uncapped', limit: 1, adaptive: false, objective: 'quadric'},
  {id: 'area-v3-cap-0.5', label: 'Area cap 0.5', short: 'Cap 0.5', limit: 0.5, adaptive: false, objective: 'quadric'},
  {id: 'area-v3-cap-0.5-adaptive', label: 'Area cap 0.5 + adaptive targets', short: 'Cap 0.5 + adaptive', limit: 0.5, adaptive: true, objective: 'quadric'},
  {id: 'area-v3-cap-0.5-topology-relaxed', label: 'Area cap 0.5 · topology relaxed (experimental)', short: 'Topology relaxed · experiment', limit: 0.5, adaptive: false, objective: 'topology_relaxed'},
  {id: 'area-v3-cap-0.5-fallback-v2-e', label: 'Area cap 0.5 · conditional topology fallback v2 (post-hoc)', short: 'Conditional fallback v2 · experiment', limit: 0.5, adaptive: false, objective: 'quadric', topology_fallback: true},
];
const names = new Map([
  ['ph_painted_wooden_shelves', 'Painted wooden shelves'],
  ['ph_metal_stool_02', 'Metal stool'],
  ['ph_grass_bermuda_01', 'Bermuda grass'],
  ['ph_dead_quiver_trunk', 'Quiver tree trunk'],
  ['ph_moon_rock_02', 'Moon rock'],
  ['ph_rock_face_02', 'Rock face'],
]);
const colors = ['#cb986d', '#74a6a0', '#85a869', '#95a967', '#bd9473', '#a88070', '#7b95bb', '#b18e70'];
const angles = [
  [0.54, 0.28], [0.64, 0.28], [0.55, 0.18], [0.60, 0.12],
  [0.62, 0.36], [0.64, 0.28], [0.68, 0.65], [0.76, 0.45],
];

function accessor(gltf, binary, index, component, type, element) {
  const a = gltf.accessors[index];
  assert.ok(a && a.componentType === component && a.type === type && !a.sparse, 'Unsupported glTF accessor');
  const view = gltf.bufferViews[a.bufferView];
  assert.equal(view.buffer, 0, 'Only the exported chain buffer is supported');
  const offset = a.byteOffset ?? 0, start = view.byteOffset ?? 0, stride = view.byteStride ?? element;
  assert.ok(Number.isSafeInteger(a.count) && a.count >= 0 && stride >= element && offset >= 0 && start >= 0);
  assert.ok(start + view.byteLength <= binary.length &&
    (a.count === 0 || offset + (a.count - 1) * stride + element <= view.byteLength), 'Accessor exceeds chain buffer');
  const out = Buffer.allocUnsafe(a.count * element);
  for (let i = 0; i < a.count; ++i)
    binary.copy(out, i * element, start + offset + i * stride, start + offset + i * stride + element);
  return out;
}

function extract(gltf, binary, lod) {
  const mesh = gltf.meshes[gltf.nodes[lod].mesh];
  assert.ok(mesh?.primitives?.length, 'Missing exported mesh');
  const positions = [], indices = [];
  let base = 0;
  for (const primitive of mesh.primitives) {
    assert.equal(primitive.mode ?? 4, 4, 'Only triangle primitives are supported');
    const p = accessor(gltf, binary, primitive.attributes.POSITION, 5126, 'VEC3', 12);
    const ix = accessor(gltf, binary, primitive.indices, 5125, 'SCALAR', 4);
    assert.equal(ix.length % 12, 0, 'Incomplete triangle');
    const adjusted = Buffer.allocUnsafe(ix.length);
    for (let i = 0; i < ix.length; i += 4) {
      const n = ix.readUInt32LE(i);
      assert.ok(n < p.length / 12 && n + base <= 0xffffffff, 'Index outside primitive');
      adjusted.writeUInt32LE(n + base, i);
    }
    for (let i = 0; i < p.length; i += 4) assert.ok(Number.isFinite(p.readFloatLE(i)), 'Nonfinite position');
    positions.push(p); indices.push(adjusted); base += p.length / 12;
  }
  return {positions: Buffer.concat(positions), indices: Buffer.concat(indices)};
}

function sourceFrame(positions) {
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 12)
    for (let axis = 0; axis < 3; ++axis) {
      const n = positions.readFloatLE(i + axis * 4);
      low[axis] = Math.min(low[axis], n); high[axis] = Math.max(high[axis], n);
    }
  const center = low.map((n, axis) => Math.fround((n + high[axis]) * 0.5));
  let radius2 = 0;
  for (let i = 0; i < positions.length; i += 12) {
    let d2 = 0;
    for (let axis = 0; axis < 3; ++axis) {
      const d = positions.readFloatLE(i + axis * 4) - center[axis];
      d2 += d * d;
    }
    radius2 = Math.max(radius2, d2);
  }
  return {center, radius: Math.sqrt(radius2) || 1};
}

function worst(lods, kind) {
  let out = {area: 0, level: 0, view: 0};
  for (let level = 1; level < lods.length; ++level) {
    const m = lods[level][kind];
    if (m.changed_area > out.area)
      out = {area: m.changed_area, level, view: m.changed_area_worst_view};
  }
  return out;
}

const pilot = await read(join(root, 'research/pilot.json'));
assert.equal(pilot.assets.length, 8, 'The board requires all eight frozen pilot assets');
const geometries = {};
const frameById = new Map(), sourceById = new Map();
const manifest = {version: 1, description: 'Actual exported chain geometry; untextured presentation views are not audit cameras', runs: []};
const runs = [];
let reference = null;
for (const spec of runSpecs) {
  const dir = join(root, 'research/runs', spec.id);
  const [meta, summary] = await Promise.all([read(join(dir, 'metadata.json')), read(join(dir, 'summary.json'))]);
  assert.equal(summary.run_sha256, meta.run_sha256, `${spec.id}: summary/run mismatch`);
  assert.equal(summary.complete, true, `${spec.id}: benchmark still incomplete`);
  assert.equal(summary.expected, 8, `${spec.id}: pilot size mismatch`);
  assert.equal(summary.completed, 8, `${spec.id}: missing asset rows`);
  assert.equal(meta.config.max_changed_area, spec.limit);
  assert.equal(meta.config.research.adaptive_targets, spec.adaptive);
  assert.equal(meta.config.objective, spec.objective);
  assert.equal(meta.config.research.topology_fallback ?? false, spec.topology_fallback ?? false);
  if (reference) {
    if (spec.topology_fallback) assert.notEqual(meta.binary_sha256, reference.binary_sha256,
      'Conditional fallback uses a separate feature build and must disclose that provenance');
    else assert.equal(meta.binary_sha256, reference.binary_sha256, 'Comparison uses different binaries');
    assert.equal(meta.protocol_sha256, reference.protocol_sha256, 'Comparison uses different protocols');
    assert.equal(meta.manifest_sha256, reference.manifest_sha256, 'Comparison uses different pilot assets');
    const comparable = config => {const copy = structuredClone(config); delete copy.max_changed_area; copy.research.adaptive_targets = false;
      if (spec.objective === 'topology_relaxed') copy.objective = reference.config.objective;
      if (spec.topology_fallback) delete copy.research.topology_fallback;
      return copy;};
    assert.deepEqual(comparable(meta.config), comparable(reference.config), 'Comparison changes another setting');
  } else reference = meta;
  const run = {id: spec.id, label: spec.label, short: spec.short, limit: spec.limit, adaptive: spec.adaptive,
    objective: spec.objective, topology_fallback: spec.topology_fallback ?? false, extra_proposals: 0,
    score: summary.score, run_sha256: meta.run_sha256, config_sha256: meta.config_sha256,
    binary_sha256: meta.binary_sha256, assets: []};
  const categoryRatios = new Map();
  const record = {id: spec.id, run_sha256: meta.run_sha256, config_sha256: meta.config_sha256,
    binary_sha256: meta.binary_sha256, assets: []};
  for (const [assetIndex, asset] of pilot.assets.entries()) {
    const sourceUrl = asset.license_evidence?.url ?? asset.source_url;
    const row = await read(join(dir, 'rows', `${asset.id}.json`));
    assert.equal(row.id, asset.id); assert.equal(row.run_sha256, meta.run_sha256);
    assert.equal(row.complete, true, `${spec.id}/${asset.id}: incomplete row`);
    assert.notEqual(row.failed, true, `${spec.id}/${asset.id}: failed row`);
    if (!categoryRatios.has(asset.category)) categoryRatios.set(asset.category, []);
    categoryRatios.get(asset.category).push(row.ratio);
    const lods = row.result?.lods;
    assert.equal(lods?.length, 8, `${spec.id}/${asset.id}: expected eight LODs`);
    const chainRatio = lods.slice(1).reduce((sum, lod) => sum + lod.triangles, 0) /
      ((lods.length - 1) * lods[0].triangles);
    assert.ok(Math.abs(chainRatio - row.ratio) < 1e-12, `${spec.id}/${asset.id}: chain ratio mismatch`);
    const extraProposals = row.result.proposal_diagnostics?.topology_fallback_proposals ?? 0;
    run.extra_proposals += extraProposals;
    const [gltfText, binary] = await Promise.all([
      readFile(join(dir, 'meshes', asset.id, 'chain.gltf')),
      readFile(join(dir, 'meshes', asset.id, 'chain.bin')),
    ]);
    const gltf = JSON.parse(gltfText.toString('utf8'));
    assert.equal(gltf.nodes.length, lods.length, 'glTF node/LOD mismatch');
    const geometryHashes = [];
    for (const [level, lod] of lods.entries()) {
      assert.ok(lod.source.complete && lod.source.passed && lod.adjacent.complete && lod.adjacent.passed,
        `${spec.id}/${asset.id}/LOD${level}: acceptance failed`);
      assert.ok(lod.source.changed_area <= spec.limit + 1e-12 && lod.adjacent.changed_area <= spec.limit + 1e-12,
        `${spec.id}/${asset.id}/LOD${level}: area limit violated`);
      assert.ok(lod.source.error_px <= lod.source_limit + 1e-12 &&
        lod.adjacent.error_px <= lod.transition_limit + 1e-12,
        `${spec.id}/${asset.id}/LOD${level}: pixel limit violated`);
      const geometry = extract(gltf, binary, level);
      assert.equal(geometry.indices.length / 12, lod.triangles, `${spec.id}/${asset.id}/LOD${level}: triangle mismatch`);
      const hash = sha(Buffer.concat([geometry.positions, geometry.indices]));
      geometryHashes.push(hash);
      geometries[hash] ??= {positions: geometry.positions.toString('base64'), indices: geometry.indices.toString('base64')};
      if (level === 0) {
        const source = sourceById.get(asset.id);
        if (source) assert.equal(hash, source, `${asset.id}: source geometry differs across runs`);
        else {sourceById.set(asset.id, hash); frameById.set(asset.id, sourceFrame(geometry.positions));}
      }
    }
    const levels = lods.map((lod, level) => ({level, triangles: lod.triangles, screen_pixels: lod.screen_pixels,
      source: {area: lod.source.changed_area, area_view: lod.source.changed_area_worst_view,
        px: lod.source.error_px, px_limit: lod.source_limit},
      adjacent: {area: lod.adjacent.changed_area, area_view: lod.adjacent.changed_area_worst_view,
        px: lod.adjacent.error_px, px_limit: lod.transition_limit},
      geometry: geometryHashes[level]}));
    const entry = {id: asset.id, name: names.get(asset.id) ?? asset.title ?? asset.source_identity,
      category: asset.category === 'stress' ? 'complex scan' : asset.category, color: colors[assetIndex],
      source_url: sourceUrl, license: asset.license,
      yaw: angles[assetIndex][0], pitch: angles[assetIndex][1], ...frameById.get(asset.id),
      fallback: row.fallback === true, source_area_worst: worst(lods, 'source'),
      adjacent_area_worst: worst(lods, 'adjacent'),
      chain_ratio: chainRatio, extra_proposals: extraProposals,
      resident_bytes: row.result.storage.total_bytes, bake_seconds: row.generation_seconds,
      peak_rss_kib: row.peak_rss_kib, levels};
    run.assets.push(entry);
    record.assets.push({id: asset.id, chain_bin_sha256: sha(binary), chain_gltf_sha256: sha(gltfText),
      geometry_sha256: geometryHashes, source_url: sourceUrl, license: asset.license});
  }
  const computedScore = 100 * (1 - [...categoryRatios.values()].reduce((sum, values) =>
    sum + values.reduce((total, value) => total + value, 0) / values.length, 0) / categoryRatios.size);
  assert.ok(Math.abs(computedScore - summary.score) < 1e-8, `${spec.id}: SCORE does not match rows`);
  runs.push(run); manifest.runs.push(record);
}

const [rotated, originalSeed] = await Promise.all([
  read(join(root, 'research/area-v3/dense-tree-cap-0.5.json')),
  read(join(root, 'research/area-v3/dense-tree-default-seed-cap-0.5.json')),
]);
const capped = runs[1], treeExport = manifest.runs[1].assets.find(asset => asset.id === 'ph_dead_quiver_trunk');
for (const check of [rotated, originalSeed]) {
  assert.equal(check.run, capped.id, 'Dense audit run mismatch');
  assert.equal(check.run_sha256, capped.run_sha256, 'Dense audit run hash mismatch');
  assert.equal(check.id, treeExport.id, 'Dense audit asset mismatch');
  assert.equal(check.chain_bin_sha256, treeExport.chain_bin_sha256, 'Dense audit geometry mismatch');
  assert.equal(check.audit.max_changed_area, capped.limit, 'Dense audit area limit mismatch');
}
const failed = rotated.lods.find(lod => lod.level === 7);
const sameSeed = originalSeed.lods.find(lod => lod.level === 7);
assert.ok(!rotated.passed && !failed.source.area_passed && failed.source.distance_passed &&
  failed.source.changed_area > capped.limit && originalSeed.passed && sameSeed.source.area_passed,
  'Unexpected independent dense audit outcome');
const independentCheck = {run: capped.id, asset_id: treeExport.id, level: 7,
  changed_area: failed.source.changed_area, area_limit: capped.limit,
  view: failed.source.changed_area_worst_view, views_evaluated: failed.source.views,
  planned_views: rotated.audit.orthographic + rotated.audit.perspective,
  supersample: rotated.audit.supersample, same_seed_area: sameSeed.source.changed_area};

const experimental = runs[3];
assert.ok(experimental.score < capped.score, 'Experimental pilot no longer has the documented SCORE regression');
const experimentalSpecs = [
  {key: 'tree', file: 'dense-tree-topology-relaxed.json', id: 'ph_dead_quiver_trunk', passed: false},
  {key: 'rock', file: 'dense-rock-topology-relaxed.json', id: 'ph_rock_face_02', passed: false},
  {key: 'bell', file: 'dense-bell-topology-relaxed.json', id: 'si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652', passed: true},
  {key: 'hatch', file: 'dense-hatch-topology-relaxed.json', id: 'si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7', passed: true},
];
const experimentalChecks = {};
for (const spec of experimentalSpecs) {
  const dense = await read(join(root, 'research/area-v3', spec.file));
  const exported = manifest.runs[3].assets.find(asset => asset.id === spec.id);
  const baseline = capped.assets.find(asset => asset.id === spec.id);
  const candidate = experimental.assets.find(asset => asset.id === spec.id);
  assert.equal(dense.run, experimental.id, `${spec.key}: dense audit run mismatch`);
  assert.equal(dense.run_sha256, experimental.run_sha256, `${spec.key}: dense audit run hash mismatch`);
  assert.equal(dense.id, exported.id, `${spec.key}: dense audit asset mismatch`);
  assert.equal(dense.chain_bin_sha256, exported.chain_bin_sha256, `${spec.key}: dense audit geometry mismatch`);
  assert.equal(dense.audit.max_changed_area, experimental.limit, `${spec.key}: dense audit area limit mismatch`);
  assert.equal(dense.passed, spec.passed, `${spec.key}: unexpected dense audit result`);
  assert.equal(dense.lods.length, 3, `${spec.key}: expected three tail levels`);
  const final = dense.lods.find(lod => lod.level === 7);
  assert.equal(final.triangles, candidate.levels[7].triangles, `${spec.key}: final triangle mismatch`);
  if (spec.passed) assert.ok(dense.lods.every(lod =>
    lod.source.complete && lod.source.passed && lod.adjacent.complete && lod.adjacent.passed),
    `${spec.key}: tail did not pass all six gates`);
  else assert.ok(final.source.changed_area > experimental.limit && !final.source.area_passed,
    `${spec.key}: expected final source area failure`);
  experimentalChecks[spec.key] = {run: experimental.id, asset_id: spec.id, passed: spec.passed,
    file: spec.file, level: 7, source_area: final.source.changed_area,
    adjacent_area: final.adjacent.changed_area, source_view: final.source.changed_area_worst_view,
    adjacent_view: final.adjacent.changed_area_worst_view, source_views_evaluated: final.source.views,
    area_limit: experimental.limit, planned_views: dense.audit.orthographic + dense.audit.perspective,
    supersample: dense.audit.supersample, cap_final_triangles: baseline.levels[7].triangles,
    experimental_final_triangles: candidate.levels[7].triangles};
}
assert.ok(!experimentalChecks.tree.passed && experimentalChecks.tree.source_area > experimental.limit &&
  !experimentalChecks.rock.passed && experimentalChecks.rock.adjacent_area > experimental.limit &&
  experimentalChecks.hatch.experimental_final_triangles < experimentalChecks.hatch.cap_final_triangles &&
  experimentalChecks.bell.experimental_final_triangles < experimentalChecks.bell.cap_final_triangles,
  'Experimental dense summary no longer matches the recorded findings');

const conditional = runs[4];
const controlId = 'area-v3-cap-0.5-fallback-v2-b';
const controlDir = join(root, 'research/runs', controlId);
const [controlMeta, controlSummary, frozenCapMeta, conditionalMeta] = await Promise.all([
  read(join(controlDir, 'metadata.json')), read(join(controlDir, 'summary.json')),
  read(join(root, 'research/runs', capped.id, 'metadata.json')),
  read(join(root, 'research/runs', conditional.id, 'metadata.json')),
]);
assert.equal(controlSummary.run_sha256, controlMeta.run_sha256, 'v2 control summary/run mismatch');
assert.ok(controlSummary.complete && controlSummary.expected === 8 && controlSummary.completed === 8,
  'v2 same-binary B control is incomplete');
assert.equal(controlMeta.binary_sha256, conditional.binary_sha256,
  'v2 B and E must use the same executable');
assert.equal(controlMeta.protocol_sha256, reference.protocol_sha256,
  'v2 B control uses a different protocol');
assert.equal(controlMeta.manifest_sha256, reference.manifest_sha256,
  'v2 B control uses different pilot assets');
assert.equal(controlMeta.config.research.topology_fallback, false,
  'v2 B control must disable the opt-in fallback');
const withoutFallback = config => {
  const copy = structuredClone(config);
  delete copy.research.topology_fallback;
  return copy;
};
assert.deepEqual(withoutFallback(controlMeta.config), withoutFallback(frozenCapMeta.config),
  'v2 B control changes another frozen B setting');
assert.deepEqual(withoutFallback(controlMeta.config), withoutFallback(conditionalMeta.config),
  'v2 B and E change another setting besides topology fallback');
assert.equal(controlSummary.score, capped.score, 'v2 B control SCORE differs from frozen B');
let controlMatchCount = 0;
for (const asset of pilot.assets) {
  const frozen = manifest.runs[1].assets.find(item => item.id === asset.id);
  const dir = join(controlDir, 'meshes', asset.id);
  assert.equal(sha(await readFile(join(dir, 'chain.bin'))), frozen.chain_bin_sha256,
    `${asset.id}: v2 B output differs from frozen B`);
  assert.equal(sha(await readFile(join(dir, 'chain.gltf'))), frozen.chain_gltf_sha256,
    `${asset.id}: v2 B glTF differs from frozen B`);
  ++controlMatchCount;
}
const conditionalChanged = new Set([
  'si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652',
  'si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7',
]);
assert.ok(conditional.score > capped.score && conditional.extra_proposals === 5,
  'Conditional fallback pilot outcome or work count changed');
let correctedOutputMatchCount = 0;
for (const asset of conditional.assets) {
  const base = manifest.runs[1].assets.find(item => item.id === asset.id);
  const current = manifest.runs[4].assets.find(item => item.id === asset.id);
  const historicalDir = join(root, 'research/runs/area-v3-cap-0.5-topology-fallback/meshes', asset.id);
  assert.equal(current.chain_bin_sha256, sha(await readFile(join(historicalDir, 'chain.bin'))),
    `${asset.id}: corrected E mesh differs from the historical E pilot`);
  assert.equal(current.chain_gltf_sha256, sha(await readFile(join(historicalDir, 'chain.gltf'))),
    `${asset.id}: corrected E glTF differs from the historical E pilot`);
  ++correctedOutputMatchCount;
  assert.equal(current.chain_bin_sha256 === base.chain_bin_sha256, !conditionalChanged.has(asset.id),
    `${asset.id}: unexpected conditional fallback output identity`);
  assert.equal(current.chain_gltf_sha256 === base.chain_gltf_sha256, !conditionalChanged.has(asset.id),
    `${asset.id}: unexpected conditional fallback glTF identity`);
  assert.equal(asset.extra_proposals, asset.id === [...conditionalChanged][0] ? 3 :
    asset.id === [...conditionalChanged][1] ? 2 : 0, `${asset.id}: unexpected extra proposal work`);
}
const conditionalDenseSpecs = [
  {key: 'bell', file: 'dense-bell-topology-fallback-v2.json', id: [...conditionalChanged][0]},
  {key: 'hatch', file: 'dense-hatch-topology-fallback-v2.json', id: [...conditionalChanged][1]},
];
const conditionalDense = {};
for (const spec of conditionalDenseSpecs) {
  const dense = await read(join(root, 'research/area-v3', spec.file));
  const exported = manifest.runs[4].assets.find(asset => asset.id === spec.id);
  const baseline = capped.assets.find(asset => asset.id === spec.id);
  const candidate = conditional.assets.find(asset => asset.id === spec.id);
  assert.equal(dense.run, conditional.id, `${spec.key}: dense audit run mismatch`);
  assert.equal(dense.run_sha256, conditional.run_sha256, `${spec.key}: dense audit run hash mismatch`);
  assert.equal(dense.id, exported.id, `${spec.key}: dense audit asset mismatch`);
  assert.equal(dense.chain_bin_sha256, exported.chain_bin_sha256, `${spec.key}: dense audit geometry mismatch`);
  assert.equal(dense.chain_gltf_sha256, exported.chain_gltf_sha256, `${spec.key}: dense audit glTF mismatch`);
  assert.equal(dense.audit.max_changed_area, conditional.limit, `${spec.key}: dense audit area limit mismatch`);
  assert.ok(dense.passed && dense.lods.length === 3 && dense.lods.every(lod =>
    lod.source.complete && lod.source.passed && lod.adjacent.complete && lod.adjacent.passed),
    `${spec.key}: conditional fallback tail did not pass six dense gates`);
  const final = dense.lods.find(lod => lod.level === 7);
  assert.equal(final.triangles, candidate.levels[7].triangles, `${spec.key}: final triangle mismatch`);
  conditionalDense[spec.key] = {asset_id: spec.id, source_area: final.source.changed_area,
    adjacent_area: final.adjacent.changed_area, area_limit: conditional.limit,
    planned_views: dense.audit.orthographic + dense.audit.perspective,
    supersample: dense.audit.supersample, cap_final_triangles: baseline.levels[7].triangles,
    conditional_final_triangles: candidate.levels[7].triangles};
}
const conditionalTreeAudit = await read(join(root, 'research/area-v3/dense-tree-topology-fallback-v2.json'));
const conditionalTreeExport = manifest.runs[4].assets.find(asset => asset.id === treeExport.id);
const conditionalTree = conditional.assets.find(asset => asset.id === treeExport.id);
assert.equal(conditionalTreeAudit.run, conditional.id, 'E tree dense audit run mismatch');
assert.equal(conditionalTreeAudit.run_sha256, conditional.run_sha256, 'E tree dense audit run hash mismatch');
assert.equal(conditionalTreeAudit.id, conditionalTreeExport.id, 'E tree dense audit asset mismatch');
assert.equal(conditionalTreeAudit.chain_bin_sha256, conditionalTreeExport.chain_bin_sha256,
  'E tree dense audit geometry mismatch');
assert.equal(conditionalTreeAudit.chain_gltf_sha256, conditionalTreeExport.chain_gltf_sha256,
  'E tree dense audit glTF mismatch');
assert.equal(conditionalTreeAudit.audit.max_changed_area, conditional.limit,
  'E tree dense audit area limit mismatch');
assert.equal(conditionalTreeAudit.lods.length, 3, 'E tree dense audit must cover the three tail levels');
const conditionalTreeFinal = conditionalTreeAudit.lods.find(lod => lod.level === 7);
assert.equal(conditionalTreeFinal.triangles, conditionalTree.levels[7].triangles,
  'E tree dense audit final triangle mismatch');
assert.ok(!conditionalTreeAudit.passed && !conditionalTreeFinal.source.area_passed &&
  conditionalTreeFinal.source.distance_passed && conditionalTreeFinal.adjacent.passed &&
  conditionalTreeFinal.source.changed_area > conditional.limit,
  'E tree dense audit must record the final source-area failure');
assert.equal(conditionalTreeFinal.source.changed_area, independentCheck.changed_area,
  'Unchanged E tree must reproduce B rotated dense failure');
const conditionalCheck = {run: conditional.id, control_run: controlId, control_score: controlSummary.score,
  control_match_count: controlMatchCount, corrected_output_match_count: correctedOutputMatchCount,
  extra_proposals: conditional.extra_proposals,
  changed_assets: [...conditionalChanged], bell: conditionalDense.bell, hatch: conditionalDense.hatch,
  tree: {asset_id: conditionalTreeExport.id, source_area: conditionalTreeFinal.source.changed_area,
    source_view: conditionalTreeFinal.source.changed_area_worst_view, area_limit: conditional.limit}};

const data = {version: 1, protocol: 'opaque geometry; fixed 512→16 px, eight audit slots, 12+4 audit views at 4×',
  pilot_ids: pilot.assets.map(asset => asset.id), runs, independent_check: independentCheck,
  experimental_checks: experimentalChecks, conditional_check: conditionalCheck, geometries};
const template = await readFile(join(output, 'template.html'), 'utf8');
assert.equal(template.split('@AREA_BOARD_DATA@').length, 2, 'Board template must have one data token');
const serialized = JSON.stringify(data).replaceAll('<', '\\u003c');
await mkdir(output, {recursive: true});
await writeFile(join(output, 'index.html'), template.replace('@AREA_BOARD_DATA@', serialized));
await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Wrote ${output}/index.html: ${runs.length} runs × ${pilot.assets.length} assets × 8 LODs; ${Object.keys(geometries).length} unique meshes`);
