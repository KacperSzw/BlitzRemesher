// One persistent process owns teacher, dataset, optimizer, refresh and audits.
// Building these arguments does not launch training.
import fs from 'node:fs';
import {read,write} from './runpod-api.mjs';
import {replayPackedDomain} from './packed-domain-proof.mjs';
export function residentArguments(directory,{model,checkpoint,memory=16384,minutes=120,finalize=10,candidateBatch=2,backend='reference',profile='coverage',states=8,updates=2048,workers=1,hiddenWidth=64,checkpointSeconds=60,episodeSeeds=false,corpus='research/corpus.json',selection='research/neural/prepared-pilot/selection.json',curriculum='research/neural/prepared-pilot/curriculum.json',meshCache=directory+'/mesh-cache'}={}){
  if(minutes!==120||!Number.isInteger(finalize)||finalize<1||finalize>30||![1,2,4,8].includes(candidateBatch)||!['reference','fused'].includes(backend)||!['coverage','attributes'].includes(profile))throw new Error('Invalid prepared learning window');
  for(const [value,max] of [[states,4096],[updates,100000],[workers,3],[checkpointSeconds,3600]])if(!Number.isInteger(value)||value<1||value>max)throw new Error('Invalid prepared pipeline setting');
  if(![64,128,256].includes(hiddenWidth)||typeof episodeSeeds!=='boolean')throw new Error('Invalid prepared policy setting');
  if(!model||(profile==='attributes'&&!checkpoint))throw new Error('Verified initialization is required');
  return [directory,'--duration-minutes',String(minutes),'--finalize-minutes',String(finalize),'--minutes',String(minutes+finalize),
    '--states',String(states),'--updates',String(updates),'--batch','512','--seed','101','--gpu-memory-mib',String(memory),
    '--raster-backend','vulkan','--vertex-storage','packed','--data-storage','compact',
    '--training-profile',profile,'--candidate-batch',String(candidateBatch),'--update-backend',backend,
    '--corpus',corpus,'--training-selection',selection,'--curriculum',curriculum,'--mesh-cache',meshCache,
    '--workers',String(workers),'--hidden-width',String(hiddenWidth),'--checkpoint-seconds',String(checkpointSeconds),'--episode-seeds',episodeSeeds?'on':'off',
    '--initialize',model,...(profile==='attributes'?['--warmstart',checkpoint]:[])];
}
export function preparedLearningOptions(plan){
  if(plan.launch!==true)throw new Error('Prepared training is not authorized for launch');
  return {minutes:plan.duration_minutes,finalize:plan.finalize_minutes,memory:plan.gpu_memory_mib,candidateBatch:plan.candidate_batch,backend:plan.update_backend,profile:plan.profile,states:plan.states_per_episode,updates:plan.updates_per_shard,workers:plan.workers,hiddenWidth:plan.hidden_width,checkpointSeconds:plan.checkpoint_seconds,episodeSeeds:true,corpus:plan.corpus,selection:plan.training_selection,curriculum:plan.curriculum};
}
export async function validatePreparedLearning(ctx,validation,plan=read('research/neural/next-training.json')){
  const options=preparedLearningOptions(plan),proof={complete:false,score:null,attempts:[]};
  ctx.phase('expanded-training-preflight');
  if(plan.packed_domain_gate===true)proof.packed_replay=await replayPackedDomain(ctx,options.memory);
  const directory=ctx.root+'/expanded-preflight',args=residentArguments(directory,{model:validation.model,...options,meshCache:ctx.root+'/mesh-cache'});
  // One complete pass at the selected states/update/worker settings. This is
  // separate from the two-hour window and cannot shorten it.
  const smoke=[directory];for(let i=1;i<args.length;i+=2)if(!['--duration-minutes','--finalize-minutes','--minutes'].includes(args[i]))smoke.push(args[i],args[i+1]);
  smoke.push('--minutes','5','--quality','off');
  const code=await ctx.execute('blitz-neural-cycle',smoke,ctx.root+'/expanded-preflight.log',5.1);
  proof.cycle=read(directory+'/report.json');write(ctx.root+'/preflight.json',proof);
  if(code!==0||!proof.cycle.complete||proof.cycle.failed_conditions_count||proof.cycle.next_condition!==read(plan.curriculum).conditions.length)throw new Error('Selected expanded training configuration failed preflight');
  for(const memory of [4096,options.memory]){
    const settings=ctx.root+'/model-audit-settings-'+memory+'.json';write(settings,{...read('research/neural/refactor-smoke.json'),profile:plan.profile,gpu_memory_mib:memory});
    const attempt={gpu_memory_mib:memory,rows:[]};
    for(const [name,model] of [['previous','/workspace/previous-model.blzn'],['initial',validation.model]]){
      const output=ctx.root+'/'+name+'-model-audit-'+memory+'.json';
      const status=await ctx.execute('blitz-neural-cycle',['--audit-model',plan.validation_selection,model,settings,output],output+'.log',4);
      const report=read(output);attempt.rows.push({name,code:status,report});
    }
    proof.attempts.push(attempt);write(ctx.root+'/preflight.json',proof);
    if(attempt.rows.every(r=>r.code===0&&r.report.complete)){proof.complete=true;proof.settings=settings;break;}
    if(!attempt.rows.every(r=>r.report.rows.every(a=>a.status==='complete'||a.workspace_limited===true)))break;
  }
  write(ctx.root+'/preflight.json',proof);if(!proof.complete)throw new Error('Large-mesh model validation failed; training remains unstarted');
  return proof;
}
export async function persistentLearningCycle(ctx,validation,{now=Date.now}={}){
  const remaining=ctx.latest-now();if(remaining<134*60000+15000)throw new Error('Full learning cycle and final comparison no longer fit');
  const plan=read('research/neural/next-training.json'),directory=ctx.root+'/resident-cycle';
  const args=residentArguments(directory,{model:validation.model,...preparedLearningOptions(plan),meshCache:ctx.root+'/mesh-cache'});
  write(ctx.root+'/resident-launch.json',{arguments:args,started:now(),optimizer_processes:1});
  ctx.phase('resident-learning-cycle');const code=await ctx.execute('blitz-neural-cycle',args,ctx.root+'/resident-cycle.log',130.2);
  const report=read(directory+'/report.json');
  if(code!==0||!report.complete||report.status!=='duration_complete'||report.learning_elapsed_ms<120*60000)throw new Error('Resident learning cycle incomplete; retain report and checkpoint');
  return report;
}
