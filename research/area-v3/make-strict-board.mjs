import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, readdir, writeFile} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

// Offline presentation of four complete, recorded pilot runs. No reduction or
// audit is performed here; every number and mesh comes from the raw run.
const root = process.argv[2] ? resolve(process.argv[2]) : resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const output = join(root, 'research/area-v3/strict-board');
const read = async path => JSON.parse(await readFile(path, 'utf8'));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const specs = [
  {cap: 3, variant: 'b', label: '3 px source cap · quadric B', short: '3 px · B'},
  {cap: 3, variant: 'e', label: '3 px source cap · conditional fallback E', short: '3 px · E'},
  {cap: 4, variant: 'b', label: '4 px source cap · quadric B', short: '4 px · B'},
  {cap: 4, variant: 'e', label: '4 px source cap · conditional fallback E', short: '4 px · E'},
].map(spec => ({...spec,
  id: `area-v3-strict-source-${spec.cap}-progressive-2-cap-0.5-${spec.variant}`}));
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
assert.equal(pilot.assets.length, 8, 'The board requires the frozen eight-asset pilot');
const geometries = {}, frameById = new Map(), sourceById = new Map();
const manifest = {version: 1, description: 'Recorded strict-scenario chain geometry; presentation views are not audit cameras', runs: []};
const runs = [];
let referenceMeta = null;
const normalized = config => {
  const copy = structuredClone(config);
  delete copy.max_lod0_delta_px;
  delete copy.research.topology_fallback;
  return copy;
};
for (const spec of specs) {
  const dir = join(root, 'research/runs', spec.id);
  const [meta, summary] = await Promise.all([read(join(dir, 'metadata.json')), read(join(dir, 'summary.json'))]);
  assert.equal(summary.run_sha256, meta.run_sha256, `${spec.id}: summary/run mismatch`);
  assert.ok(summary.complete && summary.expected === 8 && summary.completed === 8,
    `${spec.id}: incomplete eight-asset pilot`);
  assert.equal(meta.config.max_lod0_delta_px, spec.cap, `${spec.id}: source cap mismatch`);
  assert.deepEqual(meta.config.transition, [[0, 2], [1, 2]], `${spec.id}: progressive delta is not constant 2 px`);
  assert.equal(meta.config.max_changed_area, 0.5, `${spec.id}: area cap mismatch`);
  assert.equal(meta.config.research.topology_fallback, spec.variant === 'e', `${spec.id}: fallback mode mismatch`);
  assert.equal(meta.config.research.output, 'rebuild', `${spec.id}: expected forced rebuild research selection`);
  assert.equal(meta.config.objective, 'quadric');
  assert.equal(meta.config.levels, 8);
  assert.equal(meta.config.candidate_budget, 8);
  assert.deepEqual(meta.config.audit_views, {orthographic: 12, perspective: 4, seed: 2971082790});
  if (referenceMeta) {
    assert.equal(meta.binary_sha256, referenceMeta.binary_sha256, 'Strict runs use different binaries');
    assert.equal(meta.protocol_sha256, referenceMeta.protocol_sha256, 'Strict runs use different protocols');
    assert.equal(meta.manifest_sha256, referenceMeta.manifest_sha256, 'Strict runs use different pilot manifests');
    assert.deepEqual(normalized(meta.config), normalized(referenceMeta.config),
      `${spec.id}: a setting besides source cap or fallback changed`);
  } else referenceMeta = meta;
  const run = {id: spec.id, label: spec.label, short: spec.short, source_cap: spec.cap,
    area_limit: 0.5, transition: 2, topology_fallback: spec.variant === 'e',
    extra_proposals: 0, score: summary.score, run_sha256: meta.run_sha256,
    binary_sha256: meta.binary_sha256, assets: []};
  const record = {id: spec.id, run_sha256: meta.run_sha256, config_sha256: meta.config_sha256,
    binary_sha256: meta.binary_sha256, assets: []};
  const categoryRatios = new Map();
  for (const [assetIndex, asset] of pilot.assets.entries()) {
    const row = await read(join(dir, 'rows', `${asset.id}.json`));
    assert.equal(row.id, asset.id); assert.equal(row.run_sha256, meta.run_sha256);
    assert.ok(row.complete && !row.failed, `${spec.id}/${asset.id}: incomplete row`);
    const lods = row.result?.lods;
    assert.equal(lods?.length, 8, `${spec.id}/${asset.id}: expected eight LODs`);
    assert.equal(row.result.max_changed_area, 0.5);
    const chainRatio = lods.slice(1).reduce((sum, lod) => sum + lod.triangles, 0) /
      ((lods.length - 1) * lods[0].triangles);
    assert.ok(Math.abs(chainRatio - row.ratio) < 1e-12, `${spec.id}/${asset.id}: chain SCORE ratio mismatch`);
    if (!categoryRatios.has(asset.category)) categoryRatios.set(asset.category, []);
    categoryRatios.get(asset.category).push(row.ratio);
    const extra = row.result.proposal_diagnostics?.topology_fallback_proposals ?? 0;
    if (spec.variant === 'b') assert.equal(extra, 0, `${spec.id}/${asset.id}: B used fallback work`);
    run.extra_proposals += extra;
    const [gltfText, binary] = await Promise.all([
      readFile(join(dir, 'meshes', asset.id, 'chain.gltf')),
      readFile(join(dir, 'meshes', asset.id, 'chain.bin')),
    ]);
    const gltf = JSON.parse(gltfText.toString('utf8'));
    assert.equal(gltf.nodes.length, 8, 'glTF node/LOD mismatch');
    const hashes = [];
    for (const [level, lod] of lods.entries()) {
      assert.ok(lod.source.complete && lod.source.passed && lod.adjacent.complete && lod.adjacent.passed,
        `${spec.id}/${asset.id}/LOD${level}: configured acceptance failed`);
      assert.ok(lod.source.changed_area <= 0.5 + 1e-12 && lod.adjacent.changed_area <= 0.5 + 1e-12,
        `${spec.id}/${asset.id}/LOD${level}: area cap violated`);
      assert.ok(lod.source.error_px <= lod.source_limit + 1e-12 &&
        lod.adjacent.error_px <= lod.transition_limit + 1e-12,
        `${spec.id}/${asset.id}/LOD${level}: pixel limit violated`);
      if (level) {
        assert.ok(lod.source_limit <= spec.cap + 1e-12, `${spec.id}/${asset.id}/LOD${level}: source cap violated`);
        assert.ok(Math.abs(lod.transition_limit - 2) < 1e-12,
          `${spec.id}/${asset.id}/LOD${level}: progressive limit is not 2 px`);
      }
      const geometry = extract(gltf, binary, level);
      assert.equal(geometry.indices.length / 12, lod.triangles,
        `${spec.id}/${asset.id}/LOD${level}: exported triangle mismatch`);
      const hash = sha(Buffer.concat([geometry.positions, geometry.indices]));
      hashes.push(hash);
      geometries[hash] ??= {positions: geometry.positions.toString('base64'),
        indices: geometry.indices.toString('base64')};
      if (level === 0) {
        const prior = sourceById.get(asset.id);
        if (prior) assert.equal(hash, prior, `${asset.id}: source geometry differs across strict runs`);
        else {sourceById.set(asset.id, hash); frameById.set(asset.id, sourceFrame(geometry.positions));}
      }
    }
    const levels = lods.map((lod, level) => ({level, triangles: lod.triangles,
      screen_pixels: lod.screen_pixels, source_limit: lod.source_limit,
      transition_limit: lod.transition_limit,
      source: {area: lod.source.changed_area, area_view: lod.source.changed_area_worst_view,
        px: lod.source.error_px, px_limit: lod.source_limit},
      adjacent: {area: lod.adjacent.changed_area, area_view: lod.adjacent.changed_area_worst_view,
        px: lod.adjacent.error_px, px_limit: lod.transition_limit}, geometry: hashes[level]}));
    const sourceUrl = asset.license_evidence?.url ?? asset.source_url;
    run.assets.push({id: asset.id, name: names.get(asset.id) ?? asset.title ?? asset.source_identity,
      category: asset.category === 'stress' ? 'complex scan' : asset.category,
      color: colors[assetIndex], source_url: sourceUrl, license: asset.license,
      yaw: angles[assetIndex][0], pitch: angles[assetIndex][1], ...frameById.get(asset.id),
      fallback: row.fallback === true, source_area_worst: worst(lods, 'source'),
      adjacent_area_worst: worst(lods, 'adjacent'), chain_ratio: chainRatio,
      extra_proposals: extra, resident_bytes: row.result.storage.total_bytes,
      bake_seconds: row.generation_seconds, peak_rss_kib: row.peak_rss_kib, levels});
    record.assets.push({id: asset.id, chain_bin_sha256: sha(binary),
      chain_gltf_sha256: sha(gltfText), geometry_sha256: hashes});
  }
  const computedScore = 100 * (1 - [...categoryRatios.values()].reduce((sum, values) =>
    sum + values.reduce((total, value) => total + value, 0) / values.length, 0) / categoryRatios.size);
  assert.ok(Math.abs(computedScore - summary.score) < 1e-8, `${spec.id}: SCORE does not match rows`);
  runs.push(run); manifest.runs.push(record);
}

for (const capIndex of [0, 2]) {
  const baseline = runs[capIndex], experiment = runs[capIndex + 1];
  const identicalIds = [];
  for (const asset of pilot.assets) {
    const b = manifest.runs[capIndex].assets.find(item => item.id === asset.id);
    const e = manifest.runs[capIndex + 1].assets.find(item => item.id === asset.id);
    if (b.chain_bin_sha256 === e.chain_bin_sha256 && b.chain_gltf_sha256 === e.chain_gltf_sha256)
      identicalIds.push(asset.id);
  }
  experiment.baseline_score = baseline.score;
  experiment.identical_assets = identicalIds.length;
  baseline.comparison_score = experiment.score;
  baseline.identical_assets = identicalIds.length;
  baseline.identical_asset_ids = identicalIds;
}
for (const variantIndex of [0, 1]) {
  const smaller = manifest.runs[variantIndex], larger = manifest.runs[variantIndex + 2];
  const identical = smaller.assets.filter((asset, index) =>
    asset.chain_bin_sha256 === larger.assets[index].chain_bin_sha256 &&
    asset.chain_gltf_sha256 === larger.assets[index].chain_gltf_sha256).length;
  runs[variantIndex].other_cap_identical_assets = identical;
  runs[variantIndex + 2].other_cap_identical_assets = identical;
}
const denseChecks = Object.fromEntries(runs.map(run => [run.id, []]));
const denseFiles = (await readdir(join(root, 'research/area-v3')))
  .filter(file => /^dense-strict-source-[34]-progressive-2-cap-0\.5-.*\.json$/.test(file)).sort();
for (const file of denseFiles) {
  const check = await read(join(root, 'research/area-v3', file));
  const runIndex = runs.findIndex(run => run.id === check.run);
  assert.ok(runIndex >= 0, `${file}: dense audit belongs to an unknown strict run`);
  const run = runs[runIndex];
  const exported = manifest.runs[runIndex].assets.find(asset => asset.id === check.id);
  const asset = run.assets.find(asset => asset.id === check.id);
  assert.ok(exported && asset, `${file}: dense audit belongs to an unknown pilot asset`);
  assert.equal(check.run_sha256, run.run_sha256, `${file}: strict dense run SHA mismatch`);
  assert.equal(check.chain_bin_sha256, exported.chain_bin_sha256,
    `${file}: strict dense geometry mismatch`);
  assert.equal(check.chain_gltf_sha256, exported.chain_gltf_sha256,
    `${file}: strict dense glTF mismatch`);
  assert.equal(check.audit.max_changed_area, 0.5, `${file}: strict dense area cap mismatch`);
  assert.equal(check.audit.orthographic + check.audit.perspective, 706,
    `${file}: strict dense camera count mismatch`);
  assert.equal(check.audit.supersample, 8, `${file}: strict dense supersampling mismatch`);
  assert.deepEqual(check.lods.map(lod => lod.level), [5, 6, 7],
    `${file}: dense audit does not cover the three tail LODs`);
  assert.equal(check.lods.at(-1).triangles, asset.levels[7].triangles,
    `${file}: strict dense final triangle mismatch`);
  const failures = [];
  let tightest = {margin: Infinity};
  for (const lod of check.lods) for (const [kind, limit] of [
    ['source', lod.source_limit], ['adjacent', lod.transition_limit],
  ]) {
    assert.ok(kind !== 'source' || limit <= run.source_cap + 1e-12,
      `${file}: source pixel limit exceeds this run's cap`);
    assert.ok(kind !== 'adjacent' || Math.abs(limit - 2) < 1e-12,
      `${file}: adjacent pixel limit differs from 2 px`);
    const margin = limit - lod[kind].error_px;
    if (margin < tightest.margin) tightest = {level: lod.level, kind, error_px: lod[kind].error_px,
      limit, margin};
    if (!lod[kind].complete || !lod[kind].passed) failures.push({level: lod.level, kind,
      error_px: lod[kind].error_px, pixel_limit: limit,
      area: lod[kind].changed_area, area_limit: 0.5});
  }
  assert.equal(check.passed, failures.length === 0,
    `${file}: aggregate result disagrees with the six tail gates`);
  denseChecks[run.id].push({file, passed: check.passed, asset_id: check.id,
    asset_name: asset.name, seed: check.audit.seed, failures,
    final_triangles: check.lods.at(-1).triangles,
    final_source_area: check.lods.at(-1).source.changed_area,
    final_adjacent_area: check.lods.at(-1).adjacent.changed_area,
    views: check.audit.orthographic + check.audit.perspective,
    supersample: check.audit.supersample, tightest});
}
for (const capIndex of [0, 2]) {
  const baseline = runs[capIndex], experiment = runs[capIndex + 1];
  const directE = denseChecks[experiment.id].filter(check => check.seed === 3665436710 && check.passed);
  baseline.inherited_dense_assets = baseline.identical_asset_ids.filter(id =>
    directE.some(check => check.asset_id === id));
  const directB = denseChecks[baseline.id].filter(check => check.seed === 3665436710 && check.passed);
  baseline.direct_dense_first_seed_assets = [...new Set(directB.map(check => check.asset_id))].length;
  baseline.finite_first_seed_assets = new Set([
    ...baseline.inherited_dense_assets, ...directB.map(check => check.asset_id),
  ]).size;
}
const data = {version: 1, protocol: 'Opaque geometry; 512→16 px, 2 px progressive delta, source cap 3/4 px, area cap 0.5, 12+4 audit views at 4×',
  pilot_ids: pilot.assets.map(asset => asset.id), runs, dense_checks: denseChecks, geometries};
const template = await readFile(join(output, 'template.html'), 'utf8');
assert.equal(template.split('@STRICT_BOARD_DATA@').length, 2, 'Strict board template must have one data token');
const serialized = JSON.stringify(data).replaceAll('<', '\\u003c');
await mkdir(output, {recursive: true});
await writeFile(join(output, 'index.html'), template.replace('@STRICT_BOARD_DATA@', serialized));
await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Wrote ${join(output, 'index.html')}: ${runs.length} runs × ${pilot.assets.length} assets × 8 LODs; ${Object.keys(geometries).length} unique meshes`);
