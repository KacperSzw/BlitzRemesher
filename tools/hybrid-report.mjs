// Verify recorded candidate selection and exports, then summarize matched research runs.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const plans=process.argv.slice(2);if(!plans.length)throw Error('hybrid-report.mjs PLAN...');
const rows=[],groups=[];
for(const planPath of plans){const p=await read(planPath);
 for(const variant of p.variants)for(const mode of p.modes)for(const budget of p.budgets){
  const name=`${p.name}-${variant.name}-${mode}-b${budget}`,dir='research/runs/'+name,summary=await read(dir+'/summary.json'),meta=await read(dir+'/metadata.json');
  const group={name,complete:summary.complete,expected:p.assets.length,assets:[],mean_bytes:null,mean_retention:null};groups.push(group);
  for(const id of p.assets){let r;try{r=await read(dir+'/rows/'+id+'.json');}catch(e){if(e.code==='ENOENT')continue;throw e;}
   if(!r.complete||r.failed){group.assets.push({id,complete:false});continue;}
   if(r.run_sha256!==meta.run_sha256)throw Error('Run mismatch '+id);
   const d=r.result,geometry=dir+'/meshes/'+id,stat=await fs.stat(geometry+'/chain.bin');
   if(stat.size!==d.storage.total_bytes||await hash(geometry+'/chain.bin')!==r.output_sha256||await hash(geometry+'/chain.gltf')!==r.gltf_sha256)throw Error('Buffer accounting/hash mismatch '+id);
   const gltf=await read(geometry+'/chain.gltf');if(gltf.bufferViews.reduce((n,v)=>n+v.byteLength,0)!==stat.size)throw Error('Accessor payload mismatch');
   const ref=d.candidates[d.reference_candidate],selected=d.candidates[d.selected_candidate];
   if(d.lods.some((l,i)=>l.triangles!==selected.triangles[i]||l.reference_triangles!==ref.triangles[i]||!l.source.passed||!l.adjacent.passed||(i&&BigInt(l.triangles)*10000n>BigInt(ref.triangles[i])*BigInt(10000+d.triangle_overhead_bps))))throw Error('Selection contract failed '+id);
   if(selected.storage.total_bytes!==stat.size)throw Error('Selected cost mismatch');
   if(d.max_added_vertex_bytes_bps!==undefined){
    if(d.max_added_vertex_bytes_bps!==null&&BigInt(d.storage.added_vertex_bytes)>BigInt(d.added_vertex_budget_bytes))throw Error('Vertex budget exceeded '+id);
    if(d.runtime_storage?.length!==d.runtime_levels.length||d.runtime_storage.some((v,i)=>v.scheduled_index!==d.runtime_levels[i]))throw Error('Runtime storage mapping mismatch '+id);
    const added=d.runtime_storage.reduce((n,v)=>n+v.added_vertex_bytes,0),indices=d.runtime_storage.reduce((n,v)=>n+v.index_bytes,0);
    if(added!==d.storage.added_vertex_bytes||indices!==d.storage.index_bytes)throw Error('Runtime storage accounting mismatch '+id);
   }
   let last=Infinity;for(const sweep of d.selection_sweep){if(sweep.storage.total_bytes>last)throw Error('Nonmonotonic allowance');last=sweep.storage.total_bytes;}
   const total=a=>a.slice(1).reduce((n,x)=>n+x,0),triangles=d.lods.map(l=>l.triangles);
   const row={id,category:r.category,run:name,budget,origin:variant.chain??'hybrid',complete:true,triangles,reference:ref.triangles,selected_total:total(triangles),reference_total:total(ref.triangles),storage:d.storage,shared:d.lods.map(l=>l.shared_vertices),seconds:r.generation_seconds,peak_rss_kib:r.peak_rss_kib,sweep:d.selection_sweep,retention:r.ratio,baseline:{}};
   const cohort=p.assets.length===12?'validation':p.assets.length===28?'development':'matrix';
   for(const output of ['reuse','rebuild']){
    const oldName=budget===8?`vegetation-${cohort}-v3-candidate-${output}-b8`:`vegetation-quality-v3-candidate-${output}-b${budget}`;
    try{const old=await read(`research/runs/${oldName}/rows/${id}.json`);if(!old.complete)throw Error('Incomplete baseline');
     if(old.input.gltf_sha256!==r.input.gltf_sha256||old.input.binary_sha256!==r.input.binary_sha256)throw Error('Baseline input mismatch');
     for(const key of ['levels','base_pixels','last_pixels','max_lod0_delta_px','profile','candidate_budget','beam_width','search_views','audit_views','search_supersample','audit_supersample','max_supersample','transition'])if(JSON.stringify(old.config[key])!==JSON.stringify(r.config[key]))throw Error('Unmatched setting '+key+' '+id);
     const oldBytes=(await fs.stat(`research/runs/${oldName}/meshes/${id}/chain.bin`)).size;
     row.baseline[output]={run:oldName,triangles:old.result.lods.map(l=>l.triangles),total:old.result.lods.slice(1).reduce((n,l)=>n+l.triangles,0),bytes:oldBytes,seconds:old.generation_seconds,changed_area:old.result.lods.at(-1).source.changed_area};
    }catch(e){if(e.code!=='ENOENT')throw e;}
   }
   group.assets.push(row);rows.push(row);
  }
  if(group.complete){if(group.assets.length!==group.expected||group.assets.some(r=>!r.complete))throw Error('Invalid complete summary');group.mean_bytes=group.assets.reduce((n,r)=>n+r.storage.total_bytes,0)/group.expected;
   const categories=Object.groupBy(group.assets,r=>r.category);group.mean_retention=Object.values(categories).reduce((n,a)=>n+a.reduce((s,r)=>s+r.retention,0)/a.length,0)/Object.keys(categories).length;
  }
 }
}
await fs.writeFile('research/hybrid/measurements.json',JSON.stringify({version:1,scope:'opaque_card_geometry_only',scored:false,score:null,groups},null,2)+'\n');
let out='# Automatic hybrid measurements\n\nResident KiB includes unchanged LOD0, all emitted vertex attributes and u32 indices. Bake time includes generation and audits; workstation concurrency is recorded in each run. All rows below are complete, verified exports. No vegetation SCORE is assigned.\n\n| Run / asset | Final tris (reference) | Resident KiB | Added vertex KiB | Bake s | Old rebuilt final / KiB |\n|---|---:|---:|---:|---:|---:|\n';
for(const r of rows)out+=`| ${r.run} / ${r.id} | ${r.triangles.at(-1)} (${r.reference.at(-1)}) | ${(r.storage.total_bytes/1024).toFixed(1)} | ${(r.storage.added_vertex_bytes/1024).toFixed(1)} | ${r.seconds.toFixed(2)} | ${r.baseline.rebuild?`${r.baseline.rebuild.triangles.at(-1)} / ${(r.baseline.rebuild.bytes/1024).toFixed(1)}`:'—'} |\n`;
await fs.writeFile('research/hybrid/RESULTS.md',out);console.log(JSON.stringify({groups:groups.length,complete:groups.filter(g=>g.complete).length,verified_chains:rows.length}));
