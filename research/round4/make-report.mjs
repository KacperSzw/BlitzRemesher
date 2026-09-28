// Formatting only; geometry measurements come from the C++ diagnostic tools.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const read=async p=>JSON.parse(await readFile(p,'utf8'));
const root=new URL('../',import.meta.url);
const here=new URL('./',import.meta.url);
const names=['base-s512-cap8','relaxed-s512-cap8','link-s512-cap8','search-s512-cap8',
  'base-s512-cap2','search-s512-cap2','base-s128-cap8','search-s128-cap8',
  'search-s512-cap4','search-s512-cap8-n6','search-s128-cap12','search-s1024-cap8',
  'validation-base-s512-cap8','validation-search-s512-cap8','validation-final-s512-cap8',
  'final-s512-cap8','final-s512-cap2','final-s512-cap4','final-s512-cap8-n6','final-s128-cap8','final-s128-cap12','final-s1024-cap8'];
const runs={};
for(const name of names){
  const a=await read(new URL(name+'-analysis.json',here));
  const s=await read(new URL('runs/r4-'+name+'/summary.json',root));
  const m=await read(new URL('runs/r4-'+name+'/metadata.json',root));
  assert.equal(s.complete,true,'Incomplete runs have no aggregate score');
  assert.equal(a.diagnostics_complete,true,'Missing diagnostics cannot be silently omitted');
  assert.equal(a.run_sha256,s.run_sha256);assert.equal(m.run_sha256,s.run_sha256);
  runs[name]={...a,seconds:s.seconds,fallbacks:s.fallbacks,completed:s.completed,expected:s.expected,build:m.build};
}
function paired(a,b){
  const clean=c=>{const copy=structuredClone(c);delete copy.objective;return copy;};
  assert.deepEqual(clean(a.config),clean(b.config));
  assert.deepEqual(a.assets.map(x=>[x.id,x.category]),b.assets.map(x=>[x.id,x.category]));
}
const baseline=runs['base-s512-cap8'],best=runs['final-s512-cap8'];
const vb=runs['validation-base-s512-cap8'],v=runs['validation-final-s512-cap8'];
paired(baseline,best);paired(vb,v);
const f=(x,n=2)=>Number(x).toFixed(n),pct=x=>f(100*x)+'%';
const reduction=(a,b)=>100*(1-b/a);
const asset=(r,id)=>r.assets.find(x=>x.id===id);
const tree='ph_dead_quiver_trunk',tb=asset(baseline,tree),tt=asset(best,tree);
const dense=await read(new URL('dense-tree-final.json',here));
async function compareOutputs(left,right){
  const a=runs[left],b=runs[right];
  assert.deepEqual(a.assets.map(x=>x.id),b.assets.map(x=>x.id));
  const assets=[];
  for(const item of a.assets){
    const x=await read(new URL('runs/r4-'+left+'/rows/'+item.id+'.json',root));
    const y=await read(new URL('runs/r4-'+right+'/rows/'+item.id+'.json',root));
    assert.equal(x.run_sha256,a.run_sha256);assert.equal(y.run_sha256,b.run_sha256);
    assert.ok(x.output_sha256&&x.attributes_sha256&&y.output_sha256&&y.attributes_sha256);
    assets.push({id:item.id,positions_indices_equal:x.output_sha256===y.output_sha256,
      attributes_equal:x.attributes_sha256===y.attributes_sha256});
  }
  return {left,right,assets,all_equal:assets.every(a=>a.positions_indices_equal&&a.attributes_equal)};
}
const replays=[];
for(const scenario of ['s512-cap8','s512-cap2','s512-cap4','s512-cap8-n6','s128-cap8','s128-cap12','s1024-cap8']){
  paired(runs['search-'+scenario],runs['final-'+scenario]);
  replays.push(await compareOutputs('search-'+scenario,'final-'+scenario));
}
paired(runs['validation-search-s512-cap8'],v);
replays.push(await compareOutputs('validation-search-s512-cap8','validation-final-s512-cap8'));
const capIdentity=await compareOutputs('final-s512-cap4','final-s512-cap8');
await writeFile(new URL('replay-identity.json',here),JSON.stringify({replays,cap_comparison:capIdentity},null,2)+'\n');
const lines=['# Round 4 — larger screens and useful final LODs','',
  'The accepted change combines a collapse-neighbor guard with better use of the existing candidate budget. Exact duplicate runtime compaction and render-cost diagnostics are separate from the quality score.','',
  'At fixed 512→16 px settings, the development SCORE changes from **'+f(baseline.score)+' to '+f(best.score)+'**. The category-balanced final retained ratio falls '+f(reduction(baseline.final_ratio,best.final_ratio),1)+'%; the last-three-level ratio falls '+f(reduction(baseline.tail_ratio,best.tail_ratio),1)+'%. On all 20 frozen validation meshes, SCORE changes from **'+f(vb.score)+' to '+f(v.score)+'**. No held-out assets were used this round.','',
  '[Interactive board](board/index.html) · [Board image](board/board.png) · [Shareable PDF](board/board.pdf)','',
  '## Fixed-contract algorithm experiments','',
  'Eight frozen development assets; hybrid coupled rebuild, coverage, eight scheduled slots, eight proposals per slot, beam width two plus source fallback. 512→16 px, source cap 8 px, 2→3 px transition curve. Search 6+2 cameras at 2×; audit 12+4 at 4× with refinement to 8×. These are finite-camera audits.','',
  '| Variant | SCORE ↑ | Final retained ↓ | Last 3 retained ↓ | Runtime / scheduled | Seconds |',
  '|---|---:|---:|---:|---:|---:|'];
for(const [name,label] of [['base-s512-cap8','Previous implementation'],['relaxed-s512-cap8','Relax topology locks'],['link-s512-cap8','Prevent new topology locks'],['search-s512-cap8','Guard + improved search prototype'],['final-s512-cap8','Final: correct progressive ownership']]){
  const r=runs[name];paired(baseline,r);
  lines.push('| ['+label+']('+name+'-analysis.json) | '+f(r.score)+' | '+pct(r.final_ratio)+' | '+pct(r.tail_ratio)+' | '+r.runtime_lods+' / '+r.scheduled_lods+' | '+f(r.seconds,1)+' |');
}
lines.push('',
  'The reducer probe found zero initially locked edges in the stool, tree and moon rock, but many at the end of the old reduction. Removing UVs/normals did not lower those floors. Some collapses created edges with more than two incident faces, then permanently locked their endpoints. The new common-neighbor check avoids these contractions and also rejects interior contractions joining two boundary vertices. It is not a complete topology-preservation proof. See the three reducer-floor JSON files for unaudited minima and rejection counters.','',
  'Relaxing locks alone and adding the guard alone both slightly lower the main SCORE. They improve some tails but can worsen intermediate choices. These negative results are retained. The combined search change avoids redundant hybrid inputs and spends the partial final proposal round on a deeper target. Candidate and camera limits are unchanged; the old scheduler sometimes left budget unused. This remains a bounded heuristic search, without a monotonic-error or optimality assumption.','',
  'Code review then found a progressive-input ownership bug: a borrowed result from a rebuilt predecessor could be interpreted against the original source buffer. The final version discards exact unchanged proposals already represented by the incumbent and owns the predecessor streams when borrowed indices change. Both paths have a regression test. The final implementation was rerun across the scenario sweep and all 20 validation assets; prototype measurements remain archived.','',
  (replays.every(r=>r.all_equal)?'Every final replay preserved all prototype position/index and attribute hashes across seven eight-asset scenarios and the 20-asset validation cohort.':'Some final outputs differ from the prototype; use the final run for delivered results.')+' See [per-asset replay identity](replay-identity.json).','',
  '## Final meshes and redundant slots','',
  '| Asset | Previous final tris | New final tris | Previous runtime levels | New runtime levels |',
  '|---|---:|---:|---:|---:|');
for(const a of baseline.assets){
  const b=asset(best,a.id);
  lines.push('| '+a.id+' | '+a.lods.at(-1).triangles+' | '+b.lods.at(-1).triangles+' | '+a.runtime_lod_count+' | '+b.runtime_lod_count+' |');
}
lines.push('',
  'Runtime selection removes only consecutive meshes with identical position, index, normal, UV, color, tangent and material streams. It retains the first slot and its threshold for each group. The scheduled source/adjacent audit records remain intact. glTF nodes share a single mesh payload; C and C++ callers can query the retained slot indices. Compaction changes neither SCORE nor the geometry. Equal triangle counts do not justify merging different meshes. Runtime counts apply the same exact-byte rule to every archived chain; the older exporter still wrote duplicate payloads.','',
  'The tree chain changes from '+tb.lods.map(l=>l.triangles).join(' → ')+' to **'+tt.lods.map(l=>l.triangles).join(' → ')+'**. The new chain has fewer triangles but '+tt.runtime_lod_count+' distinct levels; choosing a six-slot schedule is a separate experiment below.','',
  '## Starting size, source cap and six levels','',
  'Each row is a different requested visual contract. Compare algorithm versions only within identical settings. The cap clamps the cumulative policy; increasing an inactive cap cannot loosen that policy. All rows use a 16 px final size.','',
  '| Algorithm | Start px | Requested cap px | Slots | Actual final source limit px | SCORE | Final retained | Tree final tris | Tree runtime levels |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|');
for(const name of ['base-s128-cap8','final-s128-cap8','final-s128-cap12','base-s512-cap2','final-s512-cap2','final-s512-cap4','base-s512-cap8','final-s512-cap8','final-s512-cap8-n6','final-s1024-cap8']){
  const r=runs[name],t=asset(r,tree),c=r.config;
  lines.push('| ['+(name.startsWith('base')?'Previous':'New')+']('+name+'-analysis.json) | '+c.base_pixels+' | '+c.max_lod0_delta_px+' | '+c.levels+' | '+f(t.lods.at(-1).source_limit,3)+' | '+f(r.score)+' | '+pct(r.final_ratio)+' | '+t.lods.at(-1).triangles+' | '+t.runtime_lod_count+' |');
}
lines.push('',
  'The eight-slot 512 px policy ends at 6.8905 px, so cap 8 already stops constraining the final source error. Cap 4 and cap 2 are active. A larger starting size changes both the sampling schedule and the cumulative budget; six slots also change the transition schedule. Cross-scenario SCORE differences are not algorithm wins.','',
  (capIdentity.all_equal?'Caps 4 and 8 produced identical position/index and attribute hashes on all eight development assets.':'Caps 4 and 8 selected some different meshes; see replay-identity.json.')+' The six-slot tree chain is **'+asset(runs['final-s512-cap8-n6'],tree).lods.map(l=>l.triangles).join(' → ')+'**. It trades fewer scheduled transitions for a denser final mesh than the eight-slot aggressive chain.','',
  '## Pixel and quad costs — useful diagnostics, with a caveat','',
  'Fixed sample centers at (x+0.5,y+0.5), top-left fill, source material culling, no MSAA or depth rejection. P counts covered pixel centers per primitive and Q counts touched aligned 2×2 quads per primitive. P/(4Q) is geometric lane utilization; P/unique covered pixels is pre-depth overlap. Zero-sample primitives still have setup cost. These are CPU geometry proxies, not GPU timings or measured helper invocations.','',
  '| Tree at 16 px, summed over 16 views | Previous | New |',
  '|---|---:|---:|',
  '| Triangles | '+tb.lods.at(-1).triangles+' | '+tt.lods.at(-1).triangles+' |',
  '| Projected triangles below 1 px² | '+pct(tb.lods.at(-1).render_cost.fraction_under_one_px2)+' | '+pct(tt.lods.at(-1).render_cost.fraction_under_one_px2)+' |',
  '| Projected triangles covering zero centers | '+pct(tb.lods.at(-1).render_cost.fraction_zero_samples)+' | '+pct(tt.lods.at(-1).render_cost.fraction_zero_samples)+' |',
  '| Per-primitive quads | '+tb.lods.at(-1).render_cost.primitive_quads+' | '+tt.lods.at(-1).render_cost.primitive_quads+' |',
  '| Quad lane utilization | '+pct(tb.lods.at(-1).render_cost.quad_lane_utilization)+' | '+pct(tt.lods.at(-1).render_cost.quad_lane_utilization)+' |',
  '| Union covered pixels | '+tb.lods.at(-1).render_cost.covered_pixels+' | '+tt.lods.at(-1).render_cost.covered_pixels+' |',
  '| Original source union covered pixels at 16 px | '+tb.source_render_cost_at_final.covered_pixels+' | '+tt.source_render_cost_at_final.covered_pixels+' |','',
  '**The 2-triangle result is an aggressive coverage-only choice.** Its lower quad count also reflects substantial loss of filled area. The pilot reports a worst-view changed-area fraction of '+pct(tt.lods.at(-1).source.changed_area)+'. Pixel-distance error alone does not preserve silhouette density, volume or shading on thin objects. At cap 2 the new tree keeps '+asset(runs['final-s512-cap2'],tree).lods.at(-1).triangles+' triangles. Do not interpret the cost reduction as free GPU speed at identical appearance.','',
  'The last three tree levels were additionally checked against source and predecessor with 642+64 cameras, 8× sampling and refinement up to 32×. Dense check: **'+(dense.passed?'all six gates passed':'one or more gates failed')+'**. This check does not rewrite the pilot score or establish an all-view bound. [Raw dense audit](dense-tree-final.json).','',
  '## Frozen validation split','',
  '| Variant | Assets | SCORE | Final retained | Last 3 retained | Fallbacks | Failed | Seconds |',
  '|---|---:|---:|---:|---:|---:|---:|---:|');
for(const [r,label,name] of [[vb,'Previous','validation-base-s512-cap8'],[runs['validation-search-s512-cap8'],'Search prototype','validation-search-s512-cap8'],[v,'Final','validation-final-s512-cap8']])
  lines.push('| ['+label+']('+name+'-analysis.json) | '+r.completed+'/'+r.expected+' | '+f(r.score)+' | '+pct(r.final_ratio)+' | '+pct(r.tail_ratio)+' | '+r.fallbacks+' | '+r.failed_assets+' | '+f(r.seconds,1)+' |');
lines.push('',
  'All 20 frozen validation assets remain in the denominator. Per-asset regressions are retained, not discarded. Timings include evaluation/export on a shared workstation and do not establish an isolated speedup. The main score still averages scheduled slots with equal category weight. No quad-cost term, area constraint or new normal weight was introduced.','',
  '## Acceptance and remaining work','',
  'Accept the combined guard/search change, exact runtime compaction and diagnostics. Keep the lock-relaxed objective explicitly experimental; its pilot SCORE did not improve. The stool and some intermediate chains still regress, and aircraft tails remain expensive. Coverage-area preservation and strict normal/attribute quality need their own explicit contracts and experiments. Do not silently fold these into SCORE v1.','',
  'Release, ASan/UBSan, GCC scalar, shared C ABI and installed C/C++ consumer checks are recorded in [VALIDATION.md](../../research/VALIDATION.md). New contracts cover overshared edges on a manifold handle, strided exact comparison, runtime slot/mesh mapping, proposal budget use, shared-edge rasterization, tiny triangles, culling and overlap.','',
  '## Reproduction','',
  'See [PLAN.md](PLAN.md), [ablation instructions](ablations/README.md), raw [runs](../runs/) and source/binary/configuration hashes in every run. Mesh exports are ignored by Git; regenerate them from the frozen corpus before running diagnostics.','',
  '    build/release/blitz bench research/pilot.json research/round4/configs/s512-cap8.json NEW_RUN_DIRECTORY',
  '    build/release/blitz-lod-report NEW_RUN_DIRECTORY OUTPUT-analysis.json',
  '    build/release/blitz-tail-audit NEW_RUN_DIRECTORY ph_dead_quiver_trunk OUTPUT-dense.json',
  '    node research/round4/make-report.mjs','',
  'Primary references: [quad-fragment merging](https://graphics.stanford.edu/papers/fragmerging/), [raster fill conventions](https://learn.microsoft.com/en-us/windows/win32/direct3d9/rasterization-rules), [meshoptimizer simplification options](https://github.com/zeux/meshoptimizer#simplification). Our proxy explicitly uses half-integer centers and is not a bit-exact emulation of Direct3D 9 or any GPU.','');
await writeFile(new URL('REPORT.md',here),lines.join('\n'));
const board={baseline:'r4-base-s512-cap8',
  intro:'Round 4 starts at 512 pixels and follows real meshes down to 16. Better candidate search reduces late triangles; exact repeats share one runtime level.',
  results_intro:'Preventing new topology locks helps coarse reductions. Avoiding duplicate hybrid inputs and using the full existing proposal budget improves the complete chain score.',
  comparisons:[
    {run:'r4-base-s512-cap8',label:'Previous implementation'},
    {run:'r4-relaxed-s512-cap8',label:'Relax topology locks'},
    {run:'r4-link-s512-cap8',label:'Prevent new locks'},
    {run:'r4-search-s512-cap8',label:'Guard + better search'},
    {run:'r4-final-s512-cap8',label:'Correct input ownership'}],
  validation:{score:v.score,gain:v.score-vb.score,completed:v.completed,expected:v.expected,failed:v.failed_assets},
  findings:[
    {value:'−'+f(reduction(baseline.final_ratio,best.final_ratio),1)+'%',label:'FINAL RETAINED RATIO',title:'The last levels use fewer triangles.',text:'Category-balanced final retained triangles fall from '+pct(baseline.final_ratio)+' to '+pct(best.final_ratio)+'. Last-three-level retained ratio falls '+f(reduction(baseline.tail_ratio,best.tail_ratio),1)+'%. Gates and SCORE are unchanged.'},
    {value:best.runtime_lods+'/'+best.scheduled_lods,label:'RUNTIME / AUDIT SLOTS',title:'Exact repeats share one mesh.',text:'The C/C++ API and glTF export expose compact runtime levels. Every scheduled source and transition audit stays in the manifest and in the score.'},
    {value:'2 vs 34',label:'TREE / CAP 8 vs 2',title:'A loose distance budget can lose filled area.',text:'The 2-triangle tail passes the denser 706-camera recheck but loses much of the thin trunk’s coverage. Its smaller quad cost includes that coverage loss. Shading is outside this coverage-only run.'}]};
await mkdir(new URL('board/',here),{recursive:true});
await writeFile(new URL('board/summary.json',here),JSON.stringify(board,null,2)+'\n');
const overview=names.map(name=>({name,score:runs[name].score,final_ratio:runs[name].final_ratio,tail_ratio:runs[name].tail_ratio,runtime_lods:runs[name].runtime_lods,scheduled_lods:runs[name].scheduled_lods}));
await writeFile(new URL('overview.json',here),JSON.stringify(overview,null,2)+'\n');
console.log('Wrote Round 4 report, board facts and overview from complete recorded runs.');
