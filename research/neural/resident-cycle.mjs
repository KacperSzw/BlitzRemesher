// One persistent process owns teacher, dataset, optimizer, refresh and audits.
// Building these arguments does not launch training.
import fs from 'node:fs';
import {read,write} from './runpod-api.mjs';
export function residentArguments(directory,{model,checkpoint,memory=16384,minutes=120,finalize=10,candidateBatch=1,backend='reference'}={}){
  if(minutes!==120||!Number.isInteger(finalize)||finalize<1||finalize>=minutes||![1,2,4].includes(candidateBatch)||!['reference','fused'].includes(backend))throw new Error('Invalid prepared learning window');
  if(!model||!checkpoint)throw new Error('A verified model and optimizer checkpoint are required');
  return [directory,'--duration-minutes',String(minutes),'--finalize-minutes',String(finalize),'--minutes',String(minutes),
    '--states','8','--updates','2048','--batch','512','--seed','101','--gpu-memory-mib',String(memory),
    '--raster-backend','vulkan','--vertex-storage','packed','--data-storage','compact',
    '--candidate-batch',String(candidateBatch),'--update-backend',backend,
    '--training-selection','research/neural/prepared-pilot/selection.json','--curriculum','research/neural/prepared-pilot/curriculum.json',
    '--initialize',model,'--warmstart',checkpoint];
}
export async function persistentLearningCycle(ctx,validation,{now=Date.now}={}){
  const remaining=ctx.latest-now();if(remaining<120*60000+15000)throw new Error('Full learning cycle and publication no longer fit');
  const preset=read('research/neural/prepared-pilot/run.json'),directory=ctx.root+'/resident-cycle';
  const args=residentArguments(directory,{model:validation.model,checkpoint:validation.checkpoint,...preset});
  write(ctx.root+'/resident-launch.json',{arguments:args,started:now(),optimizer_processes:1});
  ctx.phase('resident-learning-cycle');const code=await ctx.execute('blitz-neural-cycle',args,ctx.root+'/resident-cycle.log',120.2);
  const report=read(directory+'/report.json');
  if(code!==0||!report.complete||report.status!=='duration_complete')throw new Error('Resident learning cycle incomplete; retain report and checkpoint');
  return report;
}
