#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {hash,modelPayload} from './refactor-cycle.mjs';
const [directory]=process.argv.slice(2);if(!directory||fs.existsSync(directory))throw new Error('hardware-recovery.mjs FRESH_DIRECTORY');
const root=path.resolve(directory);fs.mkdirSync(root,{recursive:true});const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const report={complete:false,binary_sha256:hash('build/neural/blitz-neural-cycle'),score:null};
async function execute(name,interrupt=false){
 const run=root+'/'+name,fd=fs.openSync(run+(interrupt?'-interrupt':fs.existsSync(run)?'-resume':'-whole')+'.log','w');
 const child=spawn('build/neural/blitz-neural-cycle',[run,'--states','1','--updates','2048','--minutes','1','--gpu-memory-mib','1024','--vertex-storage','fp32','--quality','off'],{stdio:['ignore',fd,fd]});
 let killed=false;const timer=setTimeout(()=>child.kill('SIGKILL'),70000);
 const poll=interrupt?setInterval(()=>{if(!killed&&fs.existsSync(run+'/latest.json')){const latest=read(run+'/latest.json');if(latest.active_training&&latest.step<latest.active_training.target_step){report.interrupted_step=latest.step;report.interrupted_target=latest.active_training.target_step;killed=true;child.kill('SIGKILL');}}},5):null;
 try{const status=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});if(interrupt?!killed||status.signal!=='SIGKILL':status.code!==0)throw new Error('Unexpected cycle status '+JSON.stringify(status));}
 finally{clearTimeout(timer);if(poll)clearInterval(poll);fs.closeSync(fd);}
}
try{
 await execute('whole');await execute('resumed',true);await execute('resumed');
 const whole=read(root+'/whole/latest.json'),resumed=read(root+'/resumed/latest.json');
 report.identical_parameters=modelPayload(root+'/whole/'+whole.checkpoint+'/model.blzn')===modelPayload(root+'/resumed/'+resumed.checkpoint+'/model.blzn');
 report.identical_datasets=whole.datasets.length===resumed.datasets.length&&whole.datasets.every((name,i)=>hash(root+'/whole/data/'+name+'/actions.bin')===hash(root+'/resumed/data/'+resumed.datasets[i]+'/actions.bin'));
 report.same_completed_work=whole.complete&&resumed.complete&&whole.step===resumed.step&&whole.states===resumed.states;
 report.complete=report.identical_parameters&&report.identical_datasets&&report.same_completed_work;if(!report.complete)throw new Error('Interrupted learning differs from uninterrupted learning');
}catch(error){report.error=String(error);process.exitCode=1;}
finally{fs.writeFileSync(root+'/recovery.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));}
