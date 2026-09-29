// Reuse a completed independent proof only for byte-identical render exports
// and identical audit contracts. Preserve the original proof and provenance.
import fs from 'node:fs/promises';import {createHash} from 'node:crypto';import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');process.chdir(root);
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const [from='appearance-ordering',...targets]=process.argv.slice(2),pilot=await read('research/pilot.json');
for(const id of [from,...targets])if(!/^appearance-[a-z0-9-]+$/.test(id))throw Error('Invalid run ID');
const sourceMeta=await read(`research/runs/${from}/metadata.json`);
const contract=c=>{let x=structuredClone(c);delete x.research;return JSON.stringify(x);};
for(const target of targets){const meta=await read(`research/runs/${target}/metadata.json`);if(contract(meta.config)!==contract(sourceMeta.config))throw Error('Different visual contract');
 for(const asset of pilot.assets){const id=asset.id,proof=`research/appearance/audits/${from}/${id}.json`;let a;try{a=await read(proof);}catch(e){if(e.code==='ENOENT')continue;throw e;}
  if(!a.complete||a.first_level!==1||a.audit.seed!==0xA1172027)continue;
  const src=`research/runs/${from}`,dst=`research/runs/${target}`;
  if(a.run_sha256!==sourceMeta.run_sha256||a.row_sha256!==await hash(`${src}/rows/${id}.json`)||a.chain_bin_sha256!==await hash(`${src}/meshes/${id}/chain.bin`)||a.chain_gltf_sha256!==await hash(`${src}/meshes/${id}/chain.gltf`))throw Error('Original proof mismatch');
  let row;try{row=await read(`${dst}/rows/${id}.json`);}catch(e){if(e.code==='ENOENT')continue;throw e;}
  if(!row.complete||row.failed||row.run_sha256!==meta.run_sha256)continue;
  if(a.chain_bin_sha256!==await hash(`${dst}/meshes/${id}/chain.bin`)||a.chain_gltf_sha256!==await hash(`${dst}/meshes/${id}/chain.gltf`))continue;
  const dir=`research/appearance/audits/${target}`;await fs.mkdir(dir,{recursive:true});const output=`${dir}/${id}.json`;
  try{await fs.access(output);continue;}catch(e){if(e.code!=='ENOENT')throw e;}
  a.shared_proof={path:proof,sha256:await hash(proof),reason:'Identical glTF, buffer bytes, schedule, weights, and audit contract',original_seconds:a.seconds};
  a.run=target;a.run_sha256=meta.run_sha256;a.row_sha256=await hash(`${dst}/rows/${id}.json`);a.seconds=0;
  await fs.writeFile(output,JSON.stringify(a,null,2)+'\n');console.log(target+'/'+id+' uses '+from+' exact export proof');
 }
}
