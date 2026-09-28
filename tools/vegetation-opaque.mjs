// Frozen eight-asset opaque regression: old executable, unchanged defaults,
// and the experimental preset, each at the same small pilot contract.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const directory='research/vegetation/opaque';await fs.mkdir(directory,{recursive:true});
const measured=await read('research/runs/vegetation-matrix-v3-candidate-rebuild-b8/metadata.json');
const jobs=[];
for(const variant of ['baseline','legacy','candidate']){
 const binary=variant==='baseline'?'build/vegetation-baseline/blitz':'build/vegetation-final/blitz';
 const stamp={git_revision:measured.git_revision,binary_sha256:await hash(binary)};
 if(variant!=='baseline'){stamp.source_tree_sha256=measured.source_tree_sha256;stamp.source_overlay_sha256=await hash('research/vegetation/archive/source-v3.tar.gz');}
 else stamp.source_note='Executable preserved before this run, from revision '+stamp.git_revision;
 const stampPath=`${directory}/${variant}-build.json`;await fs.writeFile(stampPath,JSON.stringify(stamp,null,2)+'\n');
 for(const mode of ['reuse','rebuild']){
  const config=await read(`research/configs/pilot-${mode==='reuse'?'reuse':'coupled'}.json`);
  if(variant==='candidate')config.research={boundary_weight:10,boundary_placement:true,independent_seams:true,adaptive_targets:true};
  const path=`${directory}/${variant}-${mode}.json`;await fs.writeFile(path,JSON.stringify(config,null,2)+'\n');
  jobs.push([binary,['bench','research/pilot.json',path,`research/runs/opaque-vegetation-v3-${variant}-${mode}`,'--build-stamp',stampPath]]);
 }
}
const deadline=Date.now()+50*60*1000;let next=0;
await Promise.all(Array.from({length:2},async()=>{while(next<jobs.length){
 const minutes=(deadline-Date.now())/60000;if(minutes<=0)throw Error('Checkpoint reached; resume same command');
 const [binary,args]=jobs[next++];args.push('--minutes',String(minutes));
 await new Promise((resolve,reject)=>{const child=spawn(binary,args,{stdio:'inherit'});child.on('error',reject);child.on('close',code=>code===0?resolve():reject(Error(`${binary}: exit ${code}`)));});
}}));
