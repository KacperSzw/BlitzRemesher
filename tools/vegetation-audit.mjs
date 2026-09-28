// Independent post-selection checks. Never feed these cameras back into search.
import fs from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
const [kind='dense',binaryRoot='build/vegetation-final']=process.argv.slice(2);
if(!['dense','cost','exports'].includes(kind))throw Error('Expected dense, cost, or exports');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const plan=await read('research/vegetation/matrix.json'),jobs=[];
await fs.mkdir(`research/vegetation/${kind}`,{recursive:true});
for(const variant of ['legacy','candidate'])for(const mode of ['reuse','rebuild']){
 const run=`research/runs/vegetation-matrix-v3-${variant}-${mode}-b8`;
 if(kind==='dense')for(const id of plan.assets){
  if(variant==='legacy'&&!['ph_fern_02_clump_1','loaf_tree_1'].includes(id))continue;
  const output=`research/vegetation/dense/${variant}-${mode}-${id}.json`,binary=binaryRoot+'/blitz-tail-audit';
  let old;try{old=await read(output);}catch(e){if(e.code!=='ENOENT')throw e;}
  if(old?.binary_sha256===await hash(binary)&&old.run_sha256===(await read(run+'/metadata.json')).run_sha256&&old.row_sha256===await hash(run+'/rows/'+id+'.json')&&old.audit?.seed===0xB1172031&&old.chain_bin_sha256===await hash(run+'/meshes/'+id+'/chain.bin')&&old.chain_gltf_sha256===await hash(run+'/meshes/'+id+'/chain.gltf'))continue;
  jobs.push([binary,[run,id,output,'0xB1172031']]);
 }else if(kind==='cost')jobs.push([binaryRoot+'/blitz-lod-report',[run,`research/vegetation/cost/${variant}-${mode}.json`]]);
 else jobs.push([process.execPath,['tools/check-foliage-chains.mjs',run,`research/vegetation/exports/${variant}-${mode}.json`]]);
}
if(kind==='dense')for(const mode of ['reuse','rebuild'])for(const id of ['ph_fern_02_clump_1','loaf_tree_1']){
 const run=`research/runs/vegetation-quality-v3-candidate-${mode}-b64`,output=`research/vegetation/dense/b64-candidate-${mode}-${id}.json`,binary=binaryRoot+'/blitz-tail-audit';
 let old;try{old=await read(output);}catch(e){if(e.code!=='ENOENT')throw e;}
 if(old?.binary_sha256===await hash(binary)&&old.run_sha256===(await read(run+'/metadata.json')).run_sha256&&old.row_sha256===await hash(run+'/rows/'+id+'.json')&&old.audit?.seed===0xB1172031&&old.chain_bin_sha256===await hash(run+'/meshes/'+id+'/chain.bin')&&old.chain_gltf_sha256===await hash(run+'/meshes/'+id+'/chain.gltf'))continue;
 jobs.push([binary,[run,id,output,'0xB1172031']]);
}
if(kind==='exports')for(const cohort of ['development','validation'])for(const variant of ['legacy','candidate'])for(const mode of ['reuse','rebuild'])
 jobs.push([process.execPath,['tools/check-foliage-chains.mjs',`research/runs/vegetation-${cohort}-v3-${variant}-${mode}-b8`,`research/vegetation/exports/${cohort}-${variant}-${mode}.json`]]);
const deadline=Date.now()+50*60*1000;let next=0;
await Promise.all(Array.from({length:2},async()=>{while(next<jobs.length){
 if(Date.now()>=deadline)throw Error('50-minute checkpoint reached; resume the same command');
 const [binary,args]=jobs[next++];
 await new Promise((resolve,reject)=>{const child=spawn(binary,args,{stdio:'inherit',detached:true});
  const timer=setTimeout(()=>{try{process.kill(-child.pid,'SIGTERM');}catch{}},Math.max(1,deadline-Date.now()));
  child.on('error',e=>{clearTimeout(timer);reject(e);});child.on('close',code=>{clearTimeout(timer);code===0?resolve():reject(Error(`${binary}: exit ${code}`));});
 });
}}));
