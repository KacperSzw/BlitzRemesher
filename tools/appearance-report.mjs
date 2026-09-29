import fs from 'node:fs/promises';import {resolve} from 'node:path';import {createHash} from 'node:crypto';
const root=resolve(import.meta.dirname,'..');process.chdir(root);
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const balance=rows=>mean([...new Set(rows.map(r=>r.category))].map(c=>mean(rows.filter(r=>r.category===c).map(r=>r.ratio))));
const pilot=await read('research/pilot.json'),reference='chain-search-v3-appearance-graph-2',original='chain-search-appearance-baseline';
const runs=[];for(const [id,method] of [[reference,'baseline'],[original,'original'],...['screen','ordering','attributes','position'].map(m=>['appearance-'+m,m])]) {
 const dir=`research/runs/${id}`;let metadata;try{metadata=await read(dir+'/metadata.json');}catch(e){if(e.code==='ENOENT')continue;throw e;}
 let summary;try{summary=await read(dir+'/summary.json');}catch(e){if(e.code!=='ENOENT')throw e;}
 const rows=[];for(const asset of pilot.assets){try{const r=await read(`${dir}/rows/${asset.id}.json`);if(r.run_sha256!==metadata.run_sha256)throw Error('Row provenance mismatch');
  if(r.result)for(const [i,l] of r.result.lods.entries())if(i&&(!l.source.passed||!l.adjacent.passed||l.source.nonfinite_error||l.adjacent.nonfinite_error||l.source.error_px>l.source_limit||l.adjacent.error_px>l.transition_limit||l.source.changed_area>metadata.config.max_changed_area||l.adjacent.changed_area>metadata.config.max_changed_area))throw Error('Invalid delivered audit');
  if(r.result?.added_vertex_budget_bytes!=null&&r.result.storage.added_vertex_bytes>r.result.added_vertex_budget_bytes)throw Error('Invalid vertex storage');
  rows.push({id:r.id,category:r.category,complete:r.complete,failed:!!r.failed,fallback:!!r.fallback,ratio:r.ratio??1,triangles:r.result?.lods.map(l=>l.triangles),generation_seconds:r.generation_seconds,
   peak_rss_kib:r.peak_rss_kib,appearance_bytes:r.numerics?.appearance_peak_bytes??0,work:r.result?.candidate_evaluations,progress:r.result?.graph_search?.progress??[],
   rejections:r.result?.rejections,canonical_attributes_sha256:r.canonical_attributes_sha256,attributes_sha256:r.attributes_sha256,output_sha256:r.output_sha256});
 }catch(e){if(e.code!=='ENOENT')throw e;}}
 const complete=summary?.complete===true&&rows.length===8&&rows.every(r=>r.complete);
 const ratio=complete?balance(rows):null;if(complete&&Math.abs(100*(1-ratio)-summary.score)>1e-8)throw Error('Score mismatch');
 runs.push({id,method,dir,split:'development',preset:'appearance',metadata,complete,completed:rows.filter(r=>r.complete).length,expected:8,score:ratio==null?null:100*(1-ratio),chain_ratio:ratio,rows});
}
const ref=runs.find(r=>r.id===reference),old=runs.find(r=>r.id===original),comparisons=[];
for(const r of runs.filter(r=>r.complete&&r.id.startsWith('appearance-'))) {
 const clean=c=>{let x=structuredClone(c);delete x.research;return x;};if(JSON.stringify(clean(r.metadata.config))!==JSON.stringify(clean(ref.metadata.config)))throw Error('Cross-contract comparison');
 if(r.metadata.manifest_sha256!==ref.metadata.manifest_sha256||r.metadata.protocol_sha256!==ref.metadata.protocol_sha256)throw Error('Manifest/protocol mismatch');
 const pairs=r.rows.map(a=>{let b=ref.rows.find(b=>b.id===a.id),o=old.rows.find(b=>b.id===a.id);if(!a.failed&&!b.failed&&a.canonical_attributes_sha256!==b.canonical_attributes_sha256)throw Error('Input mismatch');
  return {id:a.id,category:a.category,gain:1-a.ratio/b.ratio,tail_gain:1-(a.triangles?.at(-1)??b.triangles.at(-1))/b.triangles.at(-1),time_ratio:a.generation_seconds==null?null:a.generation_seconds/o.generation_seconds,final_before:b.triangles.at(-1),final_after:a.triangles?.at(-1)??null};});
 const useful=pairs.filter(p=>p.tail_gain>=.25),gain=1-r.chain_ratio/ref.chain_ratio;
 comparisons.push({candidate:r.id,baseline:ref.id,time_baseline:old.id,chain_gain:gain,max_time_ratio:pairs.every(p=>Number.isFinite(p.time_ratio))?Math.max(...pairs.map(p=>p.time_ratio)):null,asset_regressions:pairs.filter(p=>p.gain< -1e-12).length,
  useful_assets:useful.length,useful_categories:new Set(useful.map(p=>p.category)).size,reduction_gate:gain>=.1&&pairs.every(p=>p.gain>=-1e-12)&&useful.length>=4&&new Set(useful.map(p=>p.category)).size===4&&!r.rows.some(x=>x.failed),pairs});
}
const report={version:1,initial_preset:'appearance',pilot_ids:pilot.assets.map(a=>a.id),runs,comparisons};await fs.writeFile('research/appearance/analysis.json',JSON.stringify(report,null,2)+'\n');
let md='# Appearance reduction measurements\n\nFrozen eight-asset pilot, identical visual/storage limits and proposal budget. Triangle gains compare against v3 appearance graph; bake ratios compare against the original appearance baseline. Individual timings are not paired performance claims.\n\n| Stage | Complete | SCORE | Chain gain | Asset regressions | Useful tails / categories | Max observed bake ratio |\n|---|---:|---:|---:|---:|---:|---:|\n';
for(const r of runs){let c=comparisons.find(c=>c.candidate===r.id);md+=`| ${r.method} | ${r.completed}/8 | ${r.score?.toFixed(3)??'—'} | ${c?(100*c.chain_gain).toFixed(2)+'%':'—'} | ${c?.asset_regressions??'—'} | ${c?c.useful_assets+' / '+c.useful_categories:'—'} | ${c&&c.max_time_ratio!=null?c.max_time_ratio.toFixed(2)+'×':'—'} |\n`;}
for(const c of comparisons){md+=`\n## ${c.candidate}\n\n| Asset | Chain gain | Final triangles | Bake ratio |\n|---|---:|---:|---:|\n`;for(const p of c.pairs)md+=`| ${p.id} | ${(100*p.gain).toFixed(2)}% | ${p.final_before}→${p.final_after} | ${p.time_ratio?.toFixed(2)??'—'}× |\n`;}
md+='\nIncomplete cohorts have no aggregate. Source fallbacks remain in the denominator. Independent qualification is recorded separately in DECISION.md.\n';await fs.writeFile('research/appearance/REPORT.md',md);console.log(JSON.stringify(comparisons.map(c=>({id:c.candidate,gain:c.chain_gain,useful:c.useful_assets,gate:c.reduction_gate}))));
