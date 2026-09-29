import fs from 'node:fs/promises';import {resolve} from 'node:path';
process.chdir(resolve(import.meta.dirname,'..'));
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),rows=[];
for(const id of (await fs.readdir('research/runs')).filter(n=>/^density-v\d+-.*-smoke-(stool|trunk)$/.test(n)).sort()) {
 const dir='research/runs/'+id,metadata=await read(dir+'/metadata.json');
 for(const file of await fs.readdir(dir+'/rows')){const r=await read(dir+'/rows/'+file);if(r.run_sha256!==metadata.run_sha256)throw Error('Smoke provenance mismatch');
  const result=r.result;if(result&&result.storage.added_vertex_bytes>result.added_vertex_budget_bytes)throw Error('Storage cap violation');
  rows.push({id,asset:r.id,complete:r.complete,run_sha256:r.run_sha256,config:metadata.config,fallback:r.fallback,triangles:result?.lods.map(l=>l.triangles),added_bytes:result?.storage.added_vertex_bytes,
   vertex_cap:result?.added_vertex_budget_bytes,seconds:r.generation_seconds,appearance_bytes:r.numerics?.appearance_peak_bytes,
   tail_probes:result?.proposals.filter(p=>p.origin==='tail_probe'&&p.pass==='baseline').map(p=>({requested:p.requested,achieved:p.achieved,gate:p.gate,uv_rejections:p.uv_rejections}))});
 }
}
await fs.writeFile('research/density/smoke.json',JSON.stringify({purpose:'Bounded development smoke; no aggregate or default-quality score',rows},null,2)+'\n');
let md='# Bounded smoke observations\n\nOne asset per run. These are diagnosis and runtime checks, not the frozen eight-asset pilot or independently qualified chains. Times are shared-workstation observations.\n\n| Run | Final triangles | Added / allowed bytes | Seconds |\n|---|---:|---:|---:|\n';
for(const r of rows)md+=`| ${r.id} | ${r.triangles?.at(-1)??'—'} | ${r.added_bytes??'—'} / ${r.vertex_cap??'—'} | ${r.seconds?.toFixed(2)??'—'} |\n`;
md+='\nThe v1–v3 variants selected no newly stored vertex buffer. Their 16,854-triangle trunk result was already observed and failed the previous independent audit. Shared source vertices in v4 admit a 17,698-triangle trunk with 9,632 added bytes; its fresh dense audit fails at LOD1–6. Stool remains unchanged. The v3 merged emitter admits denser rebuilt proposals under the storage cap, but their source appearance audits fail. Raw targets and gates are retained in [smoke.json](smoke.json).\n';
await fs.writeFile('research/density/SMOKE.md',md);
