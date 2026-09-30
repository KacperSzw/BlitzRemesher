// Validation-only entry point. Long-run launch is deliberately a separate action.
import fs from 'node:fs';
import {spawn} from 'node:child_process';
import {read,write} from './runpod-api.mjs';
const [setup,latest,minutes]=process.argv.slice(2).map(Number),started=Date.now(),root='/workspace/results/pipeline-validation';
if(![setup,latest,minutes].every(Number.isFinite)||started>=setup||![10,30].includes(minutes))throw new Error('Invalid pipeline validation deadline');
const deadline=Math.min(latest,started+minutes*60000),config=read('/workspace/validation.json');
fs.mkdirSync(root,{recursive:true});write('/workspace/results/setup-complete.json',{at:started,training_minutes:minutes,training_deadline_ms:deadline});
const report={complete:false,score:null,started,deadline,mode:config.mode,next_training_launch:false,phases:[]};let active,cancelled=false;
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{cancelled=true;active?.kill('SIGTERM');});
async function execute(name,args,log){
  if(cancelled||Date.now()+5000>=deadline)throw new Error('Validation deadline reached');
  const fd=fs.openSync(root+'/'+log,'w'),start=Date.now();active=spawn(name,args,{stdio:['ignore',fd,fd]});let hard;
  const timer=setTimeout(()=>{active?.kill('SIGTERM');hard=setTimeout(()=>active?.kill('SIGKILL'),3000);},deadline-Date.now()-5000);
  try{const code=await new Promise((resolve,reject)=>{active.once('error',reject);active.once('close',resolve);});report.phases.push({name,args,log,code,seconds:(Date.now()-start)/1000});if(code)throw new Error(log+' failed: '+code);}
  finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);active=undefined;write(root+'/report.json',report);}
}
try{
  await execute('build/neural/blitz-neural-cycle',['--replay-checkpoint','/workspace/forensic',root+'/replay','16384'],'replay.log');
  report.replay=read(root+'/replay/report.json');
  if(config.mode!=='forensic')throw new Error('Full pipeline validation is not yet prepared');
  report.complete=true;
}catch(error){report.error=String(error);process.exitCode=1;}
finally{report.finished=Date.now();write(root+'/report.json',report);write('/workspace/results/job.json',{code:process.exitCode??0,experiment:'pipeline-validation',result:root+'/report.json'});}
