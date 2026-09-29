import fs from 'node:fs/promises';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..'),dir=resolve(root,'research/chain-search');
const read=async p=>JSON.parse(await fs.readFile(resolve(root,p),'utf8'));
const report=await read('research/chain-search/analysis.json'),pilot=await read('research/pilot.json');
let timing;try{timing=await read('research/chain-search/timing.json');}catch(e){if(e.code!=='ENOENT')throw e;}
const audits=[];
for(const preset of ['coverage','appearance']) {
 const run=report.runs.find(r=>r.split==='development'&&r.preset===preset&&r.method==='graph-2');if(!run)continue;
 for(const a of pilot.assets) {
  let audit;try{audit=await read(`research/chain-search/audits/${run.id}/${a.id}.json`);}catch(e){if(e.code!=='ENOENT')throw e;}
  if(audit&&audit.run_sha256!==run.metadata.run_sha256)throw Error('Audit run mismatch');
  const row=run.rows.find(r=>r.id===a.id),failures=audit?.lods.flatMap(l=>['source','adjacent'].filter(k=>!l[k].passed).map(k=>({level:l.level,comparison:k,
   error_px:l[k].error_px,nonfinite_error:l[k].nonfinite_error,limit:l[k==='source'?'source_limit':'transition_limit'],area:l[k].changed_area,
   area_limit:l.max_changed_area,views:l[k].views,view:l[k].worst_view,area_view:l[k].changed_area_worst_view,resource_limited:l[k].resource_limited})))??[];
  audits.push({preset,id:a.id,complete:audit?.complete??false,passed:audit?.passed??false,measured_levels:audit?.lods.length??0,expected_levels:7,unchanged:row?.fallback??false,failures});
 }
}
const primary=report.comparisons.find(c=>c.split==='development'&&c.candidate.endsWith('coverage-graph-2'));
const validation=report.comparisons.find(c=>c.split==='validation'&&c.candidate.endsWith('coverage-graph-2-validation'));
const coverage=audits.filter(a=>a.preset==='coverage'),appearance=audits.filter(a=>a.preset==='appearance');
const qualified=appearance.filter(a=>a.complete&&a.passed&&!a.unchanged);
const gates={pilot:primary?.reduction_gate??false,validation:validation?.reduction_gate??false,
 time:timing?.complete&&timing?.all_repeats_gate||false,independent_coverage:coverage.length===8&&coverage.every(a=>a.complete&&a.passed),
 independent_appearance:appearance.length===8&&appearance.every(a=>a.complete&&a.passed)&&qualified.length>0};
const decision={version:1,algorithm:report.algorithm,production_default:'incumbent',promoted:false,gates,qualified_reduced_appearance_chains:qualified.map(a=>a.id),audits};
await fs.writeFile(dir+'/decision.json',JSON.stringify(decision,null,2)+'\n');
let md='# Decision: retain the graph search as an experiment\n\n';
md+='The implementation adds a bounded graph search and extends the engine manifest. Production defaults remain unchanged. The configured small-camera pilot and independent qualification are separate outcomes.\n\n';
if(primary)md+=`One graph pass reduces category-balanced chain retention by **${(100*primary.chain_gain).toFixed(2)}%**, with **${primary.asset_regressions} asset regressions** on the frozen eight-asset coverage pilot. `;
const larger=report.comparisons.find(c=>c.split==='development'&&c.candidate.endsWith('coverage-graph-4'));
if(larger)md+=`Three passes improve retention ${(100*larger.chain_gain).toFixed(2)}%, but the largest single-run per-asset bake ratio is ${larger.max_time_ratio.toFixed(2)}×. `;
md+='The appearance preset uses the normal/color/material metric; its results must not be inferred from coverage.\n\n';
md+='| Promotion gate | Result |\n|---|---|\n';for(const [key,value]of Object.entries(gates))md+=`| ${key.replaceAll('_',' ')} | ${value?'Pass':'Not passed'} |\n`;
if(validation)md+=`\nValidation (the original twenty assets): ${(100*validation.chain_gain).toFixed(2)}% lower chain retention, ${validation.asset_regressions} asset regressions.\n`;
else md+='\nValidation is not yet complete; no validation aggregate is assigned.\n';
if(timing?.complete){md+='\n## Paired timings\n\n'+timing.scope+' All output attribute hashes match the scored runs.\n\n| Asset | Three paired ratios | Median |\n|---|---|---:|\n';for(const a of timing.assets)md+=`| ${a.id} | ${a.ratios.map(x=>x.toFixed(2)+'×').join(', ')} | ${a.median_ratio.toFixed(2)}× |\n`;}
else md+='\nPaired repeat timings are incomplete; individual bake observations are not a repeatability claim.\n';
md+='\n## Independent full-chain audits\n\nAll scheduled LOD1–7 comparisons use 642+64 cameras, seed `0xA1172026`, 8× sampling refined to 32×. A comparison can fail at its first witness; `Measurement.complete=false` then means remaining views were unnecessary for rejection. The audit report is complete only when every planned comparison has a decision. Resource/time interruptions remain incomplete.\n\n';
md+='| Preset | Asset | Levels attempted | Result | Failure witnesses |\n|---|---|---:|---|---|\n';
for(const a of audits){const result=a.complete?(a.passed?(a.unchanged?'Pass · unchanged source':'Pass'):'Fail'):'Incomplete';const witness=a.failures.map(f=>f.resource_limited?`LOD${f.level} ${f.comparison}: raster resource limit after ${f.views} views`:`LOD${f.level} ${f.comparison}: ${f.resource_limited?'raster resource limit':f.nonfinite_error?'no match within metric bound':f.error_px?.toFixed(3)+' / '+f.limit+' px'}, area ${(f.area*100).toFixed(2)}%, view ${f.view}${f.resource_limited?' (resource limit)':''}`).join('; ');md+=`| ${a.preset} | ${a.id} | ${a.measured_levels}/7 | ${result} | ${witness||'—'} |\n`;}
md+=`\n**${qualified.length} reduced appearance ${qualified.length===1?'chain qualifies':'chains qualify'}** under these independent checks. Identity checks on unchanged source fallbacks are valid but provide no reduction benefit. Texture images and normal maps remain unscored. Held-out release assets have not been used.\n`;
md+='\n## Next algorithm step\n\nStart with a deterministic appearance rejection fixture and compare the coarse search screen with the full audit. The evaluator refines coverage uncertainty but does not refine a sampled appearance failure; measure whether that screen discards candidates that would pass the finer audit. Then use failing camera/attribute witnesses to target proposal density and placement. The graph improves combination of existing proposals. Three coverage passes exceed the bake allowance, while one appearance pass gives no gain. These results do not isolate screening, target density, and collapse placement as causes.\n\n';
md+='[Measurements](REPORT.md) · [Offline board](../../examples/reduction-board/index.html) · [Progress SVG](progress.svg) · [Reproduction](README.md). Earlier unsuccessful v1/v2 and smoke measurements remain in the raw run directories; they do not enter this pilot score.\n';
await fs.writeFile(dir+'/DECISION.md',md);console.log(JSON.stringify({gates,qualified:qualified.length}));
