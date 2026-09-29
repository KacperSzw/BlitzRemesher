import fs from 'node:fs/promises';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const [version='v2']=process.argv.slice(2);
const read=async p=>JSON.parse(await fs.readFile(resolve(root,p),'utf8'));
const mean=a=>a.reduce((n,v)=>n+v,0)/a.length;
const balance=(rows,field)=>mean([...new Set(rows.map(r=>r.category))].map(c=>mean(rows.filter(r=>r.category===c).map(r=>r[field]))));
const manifest=await read('research/pilot.json');
const validation=(await read('research/corpus.json')).assets.filter(a=>a.split==='validation');
const report={version:1,algorithm:version,pilot_ids:manifest.assets.map(a=>a.id),runs:[],comparisons:[]};
for(const split of ['development','validation'])for(const preset of ['coverage','appearance','strict'])for(const method of ['baseline','legacy-2','legacy-4','graph-2','graph-4','topology-4']) {
 const assets=split==='development'?manifest.assets:validation;
 const legacy=method==='baseline'||method.startsWith('legacy');
 const id=`chain-search-${legacy?'':version+'-'}${preset}-${method}${split==='validation'?'-validation':''}`,dir=`research/runs/${id}`;
 let metadata;try{metadata=await read(dir+'/metadata.json');}catch(e){if(e.code==='ENOENT')continue;throw e;}
 let summary;try{summary=await read(dir+'/summary.json');}catch(e){if(e.code!=='ENOENT')throw e;}
 const rows=[];
 for(const asset of assets){try{const row=await read(`${dir}/rows/${asset.id}.json`);if(row.run_sha256!==metadata.run_sha256)throw Error('Mismatched run row');rows.push(row);}catch(e){if(e.code!=='ENOENT')throw e;}}
 const complete=rows.length===assets.length&&rows.every(r=>r.complete)&&summary?.complete===true;
 const run={id,dir,split,preset,method,metadata,complete,completed:rows.filter(r=>r.complete).length,expected:assets.length,score:null,chain_ratio:null,rows:[]};
 for(const row of rows) {
  const r=row.result;
  if(!row.failed&&r)for(const [i,l] of r.lods.entries())if(i&&(!l.source.passed||!l.adjacent.passed||l.source.error_px>l.source_limit||l.adjacent.error_px>l.transition_limit||l.source.changed_area>metadata.config.max_changed_area||l.adjacent.changed_area>metadata.config.max_changed_area))throw Error('Invalid delivered audit: '+id+'/'+row.id);
  if(r?.added_vertex_budget_bytes!=null&&r.storage.added_vertex_bytes>r.added_vertex_budget_bytes)throw Error('Vertex budget violation');
  run.rows.push({id:row.id,category:row.category,complete:row.complete,failed:!!row.failed,failure:row.failure??null,fallback:!!row.fallback,
   ratio:row.ratio??1,final_ratio:row.final_ratio??1,last_three_ratio:row.last_three_ratio??1,
   generation_seconds:row.generation_seconds??null,peak_rss_kib:row.peak_rss_kib??null,triangles:r?.lods.map(l=>l.triangles)??[],
   work:r?.candidate_evaluations??null,audits:r?.audit_evaluations?.reduce((a,b)=>a+b,0)??null,
   bytes:r?.storage.total_bytes??null,added_bytes:r?.storage.added_vertex_bytes??null,progress:r?.graph_search?.progress??[],
   canonical_attributes_sha256:row.canonical_attributes_sha256,output_sha256:row.output_sha256,attributes_sha256:row.attributes_sha256});
 }
 if(complete){run.chain_ratio=balance(run.rows,'ratio');run.score=100*(1-run.chain_ratio);run.final_ratio=balance(run.rows,'final_ratio');run.last_three_ratio=balance(run.rows,'last_three_ratio');
  if(Math.abs(run.score-summary.score)>1e-8)throw Error('SCORE recomputation mismatch');}
 run.failures=run.rows.filter(r=>r.failed).length;run.fallbacks=run.rows.filter(r=>r.fallback).length;
 report.runs.push(run);
}
for(const run of report.runs.filter(r=>r.complete&&!['baseline','legacy-2','legacy-4'].includes(r.method))) {
 const base=report.runs.find(r=>r.split===run.split&&r.preset===run.preset&&r.method==='baseline'&&r.complete);if(!base)continue;
 const clean=c=>{const x=structuredClone(c);delete x.research;delete x.candidate_budget;return x;};
 if(JSON.stringify(clean(base.metadata.config))!==JSON.stringify(clean(run.metadata.config)))throw Error('Cross-contract comparison refused');
 if(base.metadata.manifest_sha256!==run.metadata.manifest_sha256||base.metadata.protocol_sha256!==run.metadata.protocol_sha256)throw Error('Cross-protocol comparison refused');
 const pairs=run.rows.map(r=>{const b=base.rows.find(x=>x.id===r.id);
  if(!b.failed&&!r.failed&&b.canonical_attributes_sha256!==r.canonical_attributes_sha256)throw Error('Canonical input mismatch');
  return {id:r.id,chain_gain:1-r.ratio/b.ratio,time_ratio:!b.failed&&!r.failed&&b.generation_seconds>0&&r.generation_seconds!=null?r.generation_seconds/b.generation_seconds:null,
   final_before:b.triangles.at(-1)??null,final_after:r.triangles.at(-1)??null};});
 const times=pairs.map(p=>p.time_ratio).filter(Number.isFinite);
 const gain=1-run.chain_ratio/base.chain_ratio;
 report.comparisons.push({baseline:base.id,candidate:run.id,split:run.split,chain_gain:gain,score_delta:run.score-base.score,
  max_time_ratio:times.length?Math.max(...times):null,asset_regressions:pairs.filter(p=>p.chain_gain< -1e-12).length,
  reduction_gate:gain>=.1&&pairs.every(p=>p.chain_gain>=-1e-12)&&!run.failures&&!base.failures,
  timing_scope:'Single development observations; repeat sequential paired timings before promotion',pairs});
}
const dir=resolve(root,'research/chain-search');await fs.mkdir(dir,{recursive:true});
await fs.writeFile(dir+'/analysis.json',JSON.stringify(report,null,2)+'\n');
let md='# Whole-chain reduction measurements\n\nFrozen eight-asset pilot. Primary: 512→16 px, eight levels, 3 px source cap, 2 px transitions, 50% changed area and 20% added vertex bytes. Appearance and strict coverage are separate contracts. Timings include generation and auditing on a shared workstation. Work tiers name proposal budgets, not measured speed.\n\n';
md+='| Split / preset / method | Complete | SCORE | Chain retained | Final retained | Last three retained | Failed / unchanged |\n|---|---:|---:|---:|---:|---:|---:|\n';
for(const r of report.runs)md+=`| ${r.split} / ${r.preset} / [${r.method}](../runs/${r.id}/summary.json) | ${r.completed}/${r.expected} | ${r.score?.toFixed(3)??'—'} | ${r.complete?(100*r.chain_ratio).toFixed(2)+'%':'—'} | ${r.complete?(100*r.final_ratio).toFixed(2)+'%':'—'} | ${r.complete?(100*r.last_three_ratio).toFixed(2)+'%':'—'} | ${r.failures} / ${r.fallbacks} |\n`;
for(const c of report.comparisons){md+=`\n## ${c.candidate}\n\nChain retention improves ${(100*c.chain_gain).toFixed(2)}% relative to its matched baseline. Largest observed per-asset time ratio: ${c.max_time_ratio?.toFixed(2)??'—'}×. Asset chain-total regressions: ${c.asset_regressions}. Reduction gate: ${c.reduction_gate?'pass':'fail'}. ${c.timing_scope}.\n\n| Asset | Chain gain | Final triangles | Bake time ratio |\n|---|---:|---:|---:|\n`;for(const p of c.pairs)md+=`| ${p.id} | ${(100*p.chain_gain).toFixed(2)}% | ${p.final_before??'—'}→${p.final_after??'—'} | ${p.time_ratio?.toFixed(2)??'—'}× |\n`;}
md+='\nIndependent audit and promotion decisions are recorded in DECISION.md. These tables establish configured-view outcomes; they do not establish all-view quality. Missing or incomplete runs have no aggregate score.\n';
await fs.writeFile(dir+'/REPORT.md',md);
console.log(JSON.stringify(report.comparisons.map(({candidate,chain_gain,max_time_ratio,reduction_gate})=>({candidate,chain_gain,max_time_ratio,reduction_gate})),null,2));
