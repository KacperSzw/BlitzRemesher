import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const [output='research/vegetation/measurements.json',prefix='vegetation-']=process.argv.slice(2);
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),sha=x=>createHash('sha256').update(x).digest('hex');
const collection=await read('research/foliage/manifest.json');
const names=(await fs.readdir('research/runs')).filter(n=>n.startsWith(prefix)).sort(),runs=[];
for(const name of names){
 const dir='research/runs/'+name,meta=await read(dir+'/metadata.json');let summary;try{summary=await read(dir+'/summary.json');}catch(e){if(e.code!=='ENOENT')throw e;}
 const rows=[];for(const id of meta.cohort??[]){try{const path=dir+'/rows/'+id+'.json',r=await read(path);if(r.run_sha256!==meta.run_sha256)throw Error('Mixed run identities');
  const rejected={},strategies={};for(const p of r.result?.proposals??[]){rejected[p.gate]=(rejected[p.gate]??0)+1;strategies[p.strategy]=(strategies[p.strategy]??0)+1;}
  rows.push({id,category:collection.assets.find(a=>a.id===id).category,complete:r.complete,failed:r.failed,row_sha256:sha(await fs.readFile(path)),ratio:r.ratio,tail_ratio:r.tail_ratio,final_ratio:r.final_ratio,triangles:r.result?.lods.map(l=>l.triangles),runtime_levels:r.result?.runtime_lod_count,generation_seconds:r.generation_seconds,stage_seconds:r.result?.stage_seconds,peak_rss_kib:r.peak_rss_kib,final_changed_area:r.result?.lods.at(-1).source.changed_area,duplicate_proposals:r.result?.proposal_diagnostics?.duplicate_proposals,rejected,strategies});
 }catch(e){if(e.code!=='ENOENT')throw e;}}
 const complete=!!summary?.complete&&rows.length===meta.cohort.length&&rows.every(r=>r.complete&&!r.failed);
 const categoryMean=key=>{const cats=new Map;for(const r of rows){const a=cats.get(r.category)??[];a.push(r[key]);cats.set(r.category,a);}return [...cats.values()].reduce((s,a)=>s+a.reduce((x,y)=>x+y,0)/a.length,0)/cats.size;};
 runs.push({name,run_sha256:meta.run_sha256,variant:meta.variant,output:meta.config.output,budget:meta.config.candidate_budget,complete,expected:meta.cohort.length,completed:rows.filter(r=>r.complete).length,retention:complete?{chain:categoryMean('ratio'),last_three:categoryMean('tail_ratio'),final:categoryMean('final_ratio')}:null,total_generation_seconds:rows.reduce((s,r)=>s+(r.generation_seconds??0),0),rows});
}
await fs.writeFile(output,JSON.stringify({scope:'Geometry-only research. No foliage or opaque SCORE; compare identical cohort/mode/budget/contracts.',runs},null,2)+'\n');
for(const r of runs)console.log(`${r.name}: ${r.complete?'complete':'INCOMPLETE'} chain=${r.retention?.chain.toFixed(4)} tail=${r.retention?.last_three.toFixed(4)} final=${r.rows.map(a=>a.triangles?.at(-1)).join(',')} seconds=${r.total_generation_seconds.toFixed(1)}`);
