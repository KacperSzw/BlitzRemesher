import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';

// Run from the repository root. Every benchmark resumes only matching identities.
const root='research/audit-cache', runs='research/runs/audit-cache';
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const write=(p,x)=>{fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,JSON.stringify(x,null,2)+'\n');};
const digest=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const ids=['ph_painted_wooden_shelves','ph_dead_quiver_trunk'];
const run=(exe,args)=>new Promise((resolve,reject)=>{
  const child=spawn(exe,args,{stdio:'inherit'});
  child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(new Error(`${exe} exited ${code}`)));
});
async function bench(name,mib,manifest='research/pilot.json',split='development') {
  const old=mib==='old',production=mib==='default';
  await run(`build/audit-cache-${old?'baseline':production?'production':'frozen'}/blitz`,[
    'bench',manifest,`${root}/configs/${old||production?'baseline':`cache-${mib}`}.json`,`${runs}/${name}`,
    '--split',split,'--minutes','50','--build-stamp',`${root}/builds/${old?'baseline':production?'production':'candidate'}.json`]);
}
function rows(name) {
  const dir=`${runs}/${name}`,meta=read(`${dir}/metadata.json`),summary=read(`${dir}/summary.json`);
  assert.equal(summary.complete,true,`${name}: incomplete batch has no comparison`);
  const data=fs.readdirSync(`${dir}/rows`).filter(x=>x.endsWith('.json')).map(x=>read(`${dir}/rows/${x}`));
  assert.equal(data.length,summary.expected);
  for(const r of data) {
    assert.equal(r.complete,true);assert.equal(r.failed??false,false,`${name}/${r.id}: failure`);
    assert.equal(r.run_sha256,meta.run_sha256);
    if(r.coverage_cache)assert(r.coverage_cache.peak_bytes<=(meta.config.research.coverage_cache_mib??0)*1048576);
  }
  return {meta,summary,data};
}
function compare(left,right,sameBinary=true) {
  const a=rows(left),b=rows(right);
  const config=m=>{const c=structuredClone(m.config);delete c.research.coverage_cache_mib;return c;};
  assert.deepEqual(config(a.meta),config(b.meta));
  for(const k of ['manifest_sha256','protocol_sha256','camera_sha256','compiler','cpu','backend','threads'])assert.deepEqual(a.meta[k],b.meta[k],k);
  if(sameBinary)assert.equal(a.meta.binary_sha256,b.meta.binary_sha256);
  assert.equal(a.data.length,b.data.length);assert.equal(a.summary.score,b.summary.score);
  for(const x of a.data) {
    const y=b.data.find(r=>r.id===x.id);assert(y,x.id);
    for(const k of ['source_files','canonical_attributes_sha256','output_sha256','attributes_sha256','result','numerics','ratio','final_ratio','last_three_ratio','fallback'])
      assert.deepEqual(x[k],y[k],`${left}/${right}/${x.id}: ${k}`);
  }
  return {left,right,assets:a.data.length,exact:true};
}
const median=xs=>{const a=[...xs].sort((x,y)=>x-y);return a[Math.floor(a.length/2)];};
function pilotReport() {
  const agreement=[];
  agreement.push(compare('pilot-old','pilot-1-0',false));
  for(let i=1;i<=5;i++)agreement.push(compare(`pilot-${i}-0`,`pilot-${i}-256`));
  for(let i=2;i<=5;i++)agreement.push(compare('pilot-1-0',`pilot-${i}-0`));
  const all=Array.from({length:5},(_,i)=>[rows(`pilot-${i+1}-0`),rows(`pilot-${i+1}-256`)]);
  const assets=all[0][0].data.map(r=>{
    const variants=[0,1].map(v=>all.map(pair=>pair[v].data.find(x=>x.id===r.id)));
    for(const variant of variants)for(const row of variant)
      assert.deepEqual(row.coverage_cache,variant[0].coverage_cache,`${r.id}: deterministic cache work`);
    const seconds=variants.map(v=>v.map(x=>x.generation_seconds));
    return {id:r.id,category:r.category,seconds,median_seconds:seconds.map(median),ratio:median(seconds[1])/median(seconds[0]),
      raster_seconds:variants.map(v=>median(v.map(x=>x.stage_seconds.raster))),
      distance_seconds:variants.map(v=>median(v.map(x=>x.stage_seconds.distance))),
      rasters:variants.map(v=>v[0].coverage_cache.rasters),fields:variants.map(v=>v[0].coverage_cache.fields),
      mask_hits:variants[1][0].coverage_cache.mask_hits,field_hits:variants[1][0].coverage_cache.field_hits,
      peak_cache_bytes:Math.max(...variants[1].map(x=>x.coverage_cache.peak_bytes)),
      peak_process_rss_kib:variants.map(v=>Math.max(...v.map(x=>x.peak_rss_kib)))};
  });
  const categories=[...new Set(assets.map(a=>a.category))];
  const logRatio=categories.reduce((sum,c)=>{const group=assets.filter(a=>a.category===c);return sum+group.reduce((s,a)=>s+Math.log(a.ratio),0)/group.length;},0)/categories.length;
  const ratio=Math.exp(logRatio),worst=Math.max(...assets.map(x=>x.ratio));
  const report={complete:true,agreement,assets,category_geomean_time_ratio:ratio,reduction_fraction:1-ratio,
    worst_asset_time_ratio:worst,performance_pass:ratio<=.75&&worst<=1.05,score:all[0][0].summary.score,
    median_total_generation_seconds:[0,1].map(v=>median(all.map(p=>p[v].data.reduce((sum,r)=>sum+r.generation_seconds,0)))),
    peak_cache_bytes:Math.max(...assets.map(a=>a.peak_cache_bytes)),
    timing_method:'Five complete paired repetitions, one benchmark process at a time. Order reverses each repetition. Shared workstation; no CPU pinning. No repeat excluded.'};
  write(`${root}/pilot-analysis.json`,report);return report;
}
const mode=process.argv[2];
if(mode==='smoke') {
  for(const mib of ['old',0,256])await bench(`smoke-${mib}`,mib,`${root}/smoke.json`);
  write(`${root}/smoke-agreement.json`,[compare('smoke-old','smoke-0',false),compare('smoke-0','smoke-256')]);
} else if(mode==='pilot') {
  await bench('pilot-old','old');
  for(let i=1;i<=5;i++)for(const mib of i%2?[0,256]:[256,0])await bench(`pilot-${i}-${mib}`,mib);
  const report=pilotReport();console.log(JSON.stringify({performance_pass:report.performance_pass,reduction_fraction:report.reduction_fraction,worst_asset_time_ratio:report.worst_asset_time_ratio}));
} else if(mode==='validation') {
  assert.equal(pilotReport().performance_pass,true,'Pilot signal required before validation');
  // Validation establishes equivalence. Its two workers are separate from the
  // sequential pilot timing experiment; validation timing is not a speed claim.
  await Promise.all([0,256].map(mib=>bench(`validation-${mib}`,mib,`${root}/validation.json`,'validation')));
  write(`${root}/validation-agreement.json`,compare('validation-0','validation-256'));
} else if(mode==='dense') {
  assert.equal(pilotReport().performance_pass,true,'Pilot signal required before dense follow-up');
  const comparisons=[];
  for(const id of ids) {
    const pair=[];
    for(const mib of [0,256]) {
      const output=`${root}/dense-${id}-${mib}.json`;
      await run('build/audit-cache-frozen/blitz-tail-audit',[`${runs}/pilot-1-256`,id,output,'3665436710',String(mib)]);
      pair.push(read(output));
    }
    assert.deepEqual(pair[0].lods,pair[1].lods);assert.equal(pair[0].passed,pair[1].passed);
    assert(pair[1].coverage_cache.peak_bytes<=256*1048576);
    comparisons.push({id,exact:true,passed:pair[0].passed,seconds:pair.map(r=>r.seconds),peak_cache_bytes:pair[1].coverage_cache.peak_bytes});
  }
  write(`${root}/dense-agreement.json`,comparisons);
} else if(mode==='production') {
  assert.equal(pilotReport().performance_pass,true);
  assert.equal(read(`${root}/validation-agreement.json`).exact,true);
  assert(read(`${root}/dense-agreement.json`).every(x=>x.exact));
  await bench('pilot-default','default');
  const agreement=compare('pilot-1-256','pilot-default',false);
  const explicit=rows('pilot-1-256'),defaults=rows('pilot-default');
  assert.equal(defaults.meta.config.research.coverage_cache_mib,256);
  for(const row of defaults.data)
    assert.deepEqual(row.coverage_cache,explicit.data.find(x=>x.id===row.id).coverage_cache);
  write(`${root}/production-agreement.json`,{...agreement,coverage_cache_mib:256,cache_counters_exact:true});
} else if(mode==='report') {
  const r=pilotReport();const validation=fs.existsSync(`${root}/validation-agreement.json`)?read(`${root}/validation-agreement.json`):null;
  const dense=fs.existsSync(`${root}/dense-agreement.json`)?read(`${root}/dense-agreement.json`):null;
  const production=fs.existsSync(`${root}/production-agreement.json`)?read(`${root}/production-agreement.json`):null;
  const environment=rows('pilot-1-256').meta;
  const totals=key=>[0,1].map(v=>r.assets.reduce((sum,a)=>sum+a[key][v],0));
  const f=x=>x.toFixed(2);
  let text=`# Exact coverage audit caching\n\nThe frozen production automatic coverage pilot completed five paired repetitions. Every output stream, audit record, rejection count, candidate selection, and numerical counter agrees with uncached generation. Cache-disabled output also agrees with the pre-change executable on all eight assets.\n\nCategory-balanced geometric mean generation time decreased **${f(r.reduction_fraction*100)}%**. Median total generation time was **${f(r.median_total_generation_seconds[0])} → ${f(r.median_total_generation_seconds[1])} seconds**. The largest per-asset time ratio was ${f(r.worst_asset_time_ratio)}. The 25% improvement / 5% maximum slowdown pilot gate **${r.performance_pass?'passed':'failed'}**. Peak charged cache storage was ${f(r.peak_cache_bytes/1048576)} MiB, within the 256 MiB allowance.\n\n| Asset | Uncached median s | Cached median s | Time reduction | Distance fields before → after | Cache peak MiB |\n|---|---:|---:|---:|---:|---:|\n`;
  for(const a of r.assets)text+=`| ${a.id} | ${f(a.median_seconds[0])} | ${f(a.median_seconds[1])} | ${f((1-a.ratio)*100)}% | ${a.fields.join(' → ')} | ${f(a.peak_cache_bytes/1048576)} |\n`;
  text+=`\nPer pilot run, raster builds decreased ${totals('rasters').join(' → ')} and distance-field builds decreased ${totals('fields').join(' → ')}. Outputs and cache-work counters repeat exactly across all five pairs.\n`;
  text+=`\nMeasured with ${environment.compiler}, ${environment.backend.toUpperCase()}, ${environment.threads} benchmark worker, on ${environment.cpu.replace(/^.*?:\s*/,'')}. The pinned Nix environment and complete settings are recorded in each run's metadata.\n`;
  text+=`\n## Contract and measurement\n\nThe scenario uses eight scheduled levels from 512 to 16 pixels, source cap 3 px, adjacent cap 2 px, coverage-area cap 0.5, eight proposals, beam width two, automatic storage and a 5% triangle allowance. Search cameras are 6+2 at 2×; audit cameras are 12+4 at 4×, refining to 8×. It is a small-camera development scenario, not the default quality audit. SCORE remains ${r.score.toFixed(4)} in both variants; the improvement is bake time.\n\n${r.timing_method} Raw per-asset times and ranges are in [pilot-analysis.json](pilot-analysis.json). The existing benchmark records process RSS as a process high-water mark, not per-asset live memory. Cache accounting includes requested entry-array bytes and retained vector capacities, including overlapping arrays during growth; allocator bookkeeping is reflected only in process RSS. The 256 MiB allowance is additional cache storage, not a whole-process memory limit.\n\n## Validation and dense views\n\n`;
  if(validation) {
    const checked=rows('validation-256');
    const peakCache=Math.max(...checked.data.map(x=>x.coverage_cache.peak_bytes))/1048576;
    const peakRss=Math.max(...checked.data.map(x=>x.peak_rss_kib))/1024;
    text+=`All ${validation.assets} frozen validation assets have exact cached/uncached agreement, with zero failures and zero unchanged-chain fallbacks. Cached validation peaked at ${f(peakCache)} MiB of charged cache and ${f(peakRss)} MiB process RSS. [Comparison](validation-agreement.json). Validation used two concurrent workers; its timings do not enter the pilot speed claim.\n\n`;
  } else text+='Validation has not completed.\n\n';
  if(dense) {
    text+='Independent tail checks use 642+64 cameras, rotation seed 3665436710, 8× sampling and refinement to 32× on LOD5–7. Cached and uncached measurements agree exactly. These are finite camera checks, not an all-view bound. Dense timings below are single pairs, separate from repeated generation timings.\n\n| Asset | Audit quality | Uncached s | Cached s | Cache peak MiB |\n|---|---|---:|---:|---:|\n';
    for(const a of dense)text+=`| ${a.id} | ${a.passed?'pass':'FAIL'} | ${f(a.seconds[0])} | ${f(a.seconds[1])} | ${f(a.peak_cache_bytes/1048576)} |\n`;
  }
  text+='\n## Regression checks and default\n\nThe corrected implementation passed all 9 release and 9 ASan/UBSan CTest targets before the timing experiment. Tests exercise bounded admission, cache lifetimes, disabled/tiny/full allowances, refinement, cancellation, allocation failure and exact measurements. A mutable-callback regression rejected the first implementation; its raw runs remain in [v1](v1/README.md) and are excluded from this result.\n\n';
  text+=production?`After the gates passed, coverage caching was enabled by default with a 256 MiB allowance. All ${production.assets} pilot assets were replayed using a configuration that omits the cache key: results and cache counters exactly match the explicitly enabled measured binary. [Default replay](production-agreement.json). The final default-enabled source passed all 9 [release](checks/release-final-ctest.log) and 9 [ASan/UBSan](checks/sanitize-final-ctest.log) targets.\n\n`:'Default enablement and its final replay have not completed.\n\n';
  text+='Set `research.coverage_cache_mib` to an integer from 0 to 256 in C++/CLI settings; zero disables caching. Only coverage-profile generation uses the cache. C ABI 4 is preserved; C++ clients must rebuild for the changed settings and statistics layouts. This experiment establishes speed for the stated coverage scenario; normal/attribute and full default-camera speed are unmeasured.\n\n## Reproduction and provenance\n\nRun `node research/audit-cache/run.mjs smoke`, then `pilot`, `validation`, `dense`, `production`, and `report`. The runner requires the frozen executables named in its source and validates exact agreement before reporting performance. Every batch has a 50-minute limit; incomplete batches receive no aggregate comparison. No held-out assets were used.\n\nThe baseline is Git revision 5c0c62386b212f573d2cae07bf8e33f73dfa9052. [Build stamps](builds/) identify the executables. `builds/candidate-source.tar.gz` contains the measured source snapshot; `builds/production-source.tar.gz` contains the final default-enabled source. Overlay each archive onto a separate checkout of the baseline revision, which supplies the unchanged vendored dependencies, then build with the pinned Nix shell and CMake Release. The two experiment configurations differ only in `research.coverage_cache_mib`. [Raw runs](../runs/audit-cache/) retain all rows, summaries and metadata.\n';
  fs.writeFileSync(`${root}/REPORT.md`,text);
  write(`${root}/artifacts.json`,Object.fromEntries(['run.mjs','builds/baseline.json','builds/candidate.json','builds/candidate-source.tar.gz','configs/baseline.json','configs/cache-0.json','configs/cache-256.json',...(production?['builds/production.json','builds/production-source.tar.gz']:[])].map(p=>[p,digest(`${root}/${p}`)])));
} else throw new Error('Choose smoke, pilot, validation, dense, production, or report');
