import fs from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const root='research/hybrid/dense';await fs.mkdir(root,{recursive:true});
const base=await read('research/vegetation/matrix.json');
const jobs=base.assets.map(id=>({id,run:'hybrid-development-v5-mixed-auto-b8',name:'mixed-b8-'+id}));
for(const b of [32,64])for(const id of ['ph_fern_02_clump_1','loaf_tree_1'])jobs.push({id,run:`hybrid-quality-v5-mixed-auto-b${b}`,name:`mixed-b${b}-${id}`});
const deadline=Date.now()+50*60*1000;let next=0;
const binary='build/hybrid/blitz-tail-audit',binaryHash=createHash('sha256').update(await fs.readFile(binary)).digest('hex');
await Promise.all(Array.from({length:2},async()=>{while(next<jobs.length){if(Date.now()>deadline)throw Error('Checkpoint reached; resume audits');const j=jobs[next++],out=root+'/'+j.name+'.json';
 const row=await read(`research/runs/${j.run}/rows/${j.id}.json`);
 try{const old=await read(out);if(old.run_sha256===row.run_sha256&&old.chain_bin_sha256===row.output_sha256&&old.chain_gltf_sha256===row.gltf_sha256&&old.audit.seed===0xB1172032&&old.audit_binary_sha256===binaryHash){console.log(j.name+' cached '+old.passed);continue;}throw Error('Audit identity changed '+out);}catch(e){if(e.code!=='ENOENT')throw e;}
 const code=await new Promise((resolve,reject)=>{const child=spawn(binary,['research/runs/'+j.run,j.id,out,'0xB1172032'],{stdio:'inherit'});const timer=setTimeout(()=>child.kill('SIGTERM'),Math.max(1,deadline-Date.now()));child.on('error',reject);child.on('close',code=>{clearTimeout(timer);resolve(code);});});
 if(code!==0&&code!==2)throw Error('Audit execution failed '+code);
 const d=await read(out);d.audit_binary_sha256=binaryHash;await fs.writeFile(out,JSON.stringify(d,null,2)+'\n');console.log(JSON.stringify({audit:j.name,passed:d.passed}));
}}));
