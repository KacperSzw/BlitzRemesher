// Bounded remote validation only. This never starts a two-hour training run.
import fs from 'node:fs';
import {spawn} from 'node:child_process';
import {read,write} from './runpod-api.mjs';
const [setup,latest,minutes]=process.argv.slice(2).map(Number),started=Date.now(),root='/workspace/results/hardware-validation';
if(![setup,latest,minutes].every(Number.isFinite)||started>=setup||minutes!==10)throw new Error('Invalid hardware validation deadline');
const deadline=Math.min(latest,started+minutes*60000);fs.mkdirSync(root,{recursive:true});write('/workspace/results/setup-complete.json',{at:started,training_minutes:minutes,training_deadline_ms:deadline});
let active,cancelled=false;for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{cancelled=true;active?.kill('SIGTERM');});
const report={complete:false,score:null,started,deadline,phases:[]};
async function execute(name,args,log,{validation=false,maximum=120000}={}){
  if(cancelled||Date.now()+5000>=deadline)throw new Error('Validation deadline or cancellation');
  const fd=fs.openSync(root+'/'+log,'w'),begin=Date.now(),child=spawn(name,args,{stdio:['ignore',fd,fd],env:{...process.env,...(validation?{VK_INSTANCE_LAYERS:'VK_LAYER_KHRONOS_validation'}:{})}});active=child;let hard;
  const timer=setTimeout(()=>{child.kill('SIGTERM');hard=setTimeout(()=>child.kill('SIGKILL'),5000);},Math.min(maximum,deadline-Date.now()-5000));
  try{const code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',resolve);});report.phases.push({name,args,log,code,seconds:(Date.now()-begin)/1000});if(code!==0)throw new Error(name+' failed: '+code);
    if(validation&&/Validation Error|VUID-|SYNC-HAZARD|was not found/.test(fs.readFileSync(root+'/'+log,'utf8')))throw new Error('Vulkan validation reported an error or missing layer');
  }finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);active=undefined;write(root+'/report.json',report);}
}
try{
  await execute('build/neural/blitz-neural-vulkan-tests',[],'vulkan-validation.log',{validation:true});
  await execute('/usr/local/cuda/bin/compute-sanitizer',['--tool','memcheck','--error-exitcode','1','build/neural/blitz-neural-predicate-tests'],'predicate-memcheck.log');
  await execute('/usr/local/cuda/bin/compute-sanitizer',['--tool','memcheck','--error-exitcode','1','build/neural/blitz-neural-cycle','--check'],'resident-memcheck.log');
  await execute('build/neural/blitz-neural-hardware-profile',['ph_painted_wooden_bench',root+'/raster.json','128'],'raster.log');
  await execute('build/neural/blitz-neural-cycle',[root+'/cycle','--states','2','--updates','512','--minutes','3'],'cycle.log',{maximum:190000});
  report.cycle=read(root+'/cycle/latest.json');if(!report.cycle.complete)throw new Error('Incomplete remote learning cycle');
  await execute('build/neural/blitz-neural-action-train',['--replay',root+'/cycle/'+report.cycle.checkpoint,root+'/replay.json'],'replay.log');
  report.replay=read(root+'/replay.json');if(!report.replay.passed)throw new Error('Remote checkpoint replay failed');
  await execute('build/neural/blitz-neural-action-train',['--replay','research/neural/evidence/hardware-local/replay',root+'/cross-device-replay.json'],'cross-device-replay.log');
  report.cross_device_replay=read(root+'/cross-device-replay.json');if(!report.cross_device_replay.passed)throw new Error('Local checkpoint failed cross-device replay');report.complete=true;
}catch(error){report.error=String(error);process.exitCode=1;}
finally{report.finished=Date.now();write(root+'/report.json',report);write('/workspace/results/job.json',{code:process.exitCode??0,experiment:'hardware-validation',result:root+'/report.json'});}
