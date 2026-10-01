// Validation-only entry point. Long-run launch is deliberately a separate action.
import fs from 'node:fs';
import {spawn} from 'node:child_process';
import {read,write} from './runpod-api.mjs';
import {replayPackedDomain,soakPackedDomain} from './packed-domain-proof.mjs';
process.env.PATH='/usr/local/cuda/bin:'+process.env.PATH;
const [setup,latest,minutes]=process.argv.slice(2).map(Number),started=Date.now(),root='/workspace/results/pipeline-validation';
if(![setup,latest,minutes].every(Number.isFinite)||started>=setup||![10,30].includes(minutes))throw new Error('Invalid pipeline validation deadline');
const deadline=Math.min(latest,started+minutes*60000),config=read('/workspace/validation.json');
fs.mkdirSync(root,{recursive:true});write('/workspace/results/setup-complete.json',{at:started,training_minutes:minutes,training_deadline_ms:deadline});
const report={complete:false,score:null,started,deadline,mode:config.mode,next_training_launch:false,phases:[]};let active,cancelled=false;
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{cancelled=true;active?.kill('SIGTERM');});
async function execute(name,args,log,{maximum=300000,allowFailure=false,environment={}}={}){
  if(cancelled||Date.now()+5000>=deadline)throw new Error('Validation deadline reached');
  const fd=fs.openSync(root+'/'+log,'w'),start=Date.now();active=spawn(name,args,{stdio:['ignore',fd,fd],env:{...process.env,...environment}});let hard;
  const timer=setTimeout(()=>{active?.kill('SIGTERM');hard=setTimeout(()=>active?.kill('SIGKILL'),3000);},Math.min(maximum,deadline-Date.now()-5000));
  try{const code=await new Promise((resolve,reject)=>{active.once('error',reject);active.once('close',resolve);});report.phases.push({name,args,log,code,seconds:(Date.now()-start)/1000});if(code&&!allowFailure)throw new Error(log+' failed: '+code);return code;}
  finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);active=undefined;write(root+'/report.json',report);}
}
const cycle='build/neural/blitz-neural-cycle';
try{
  await execute(cycle,['--diagnose-checkpoint','tests/fixtures/neural-cancellation',root+'/cancellation.json'],'cancellation.log');
  report.cancellation=read(root+'/cancellation.json');
  await execute(cycle,['--replay-checkpoint','/workspace/forensic',root+'/replay','16384'],'replay.log');
  report.replay=read(root+'/replay/report.json');
  if(config.mode==='packed-domain'){
    await execute('/usr/local/cuda/bin/compute-sanitizer',['--tool','memcheck','--error-exitcode','1','build/neural/blitz-neural-action-gpu-tests'],'action-memcheck.log');
    await execute('/usr/local/cuda/bin/compute-sanitizer',['--tool','memcheck','--error-exitcode','1','build/neural/blitz-neural-vulkan-tests','--memcheck'],'vulkan-memcheck.log');
    await execute('build/neural/blitz-neural-vulkan-tests',[],'vulkan-validation.log',{environment:{VK_INSTANCE_LAYERS:'VK_LAYER_KHRONOS_validation'}});
    if(/Validation Error|VUID-|SYNC-HAZARD|was not found/.test(fs.readFileSync(root+'/vulkan-validation.log','utf8')))throw new Error('Vulkan validation layer failure');
    const context={root,execute:(name,args,log,minutes)=>execute('build/neural/'+name,args,log.slice(root.length+1),{maximum:minutes*60000})};
    report.packed_replay=await replayPackedDomain(context);report.packed_soak=await soakPackedDomain(context);report.complete=true;
  }else if(config.mode==='pipeline'){
    await execute('build/neural/blitz-neural-vulkan-tests',[],'vulkan-validation.log',{environment:{VK_INSTANCE_LAYERS:'VK_LAYER_KHRONOS_validation'}});
    if(/Validation Error|VUID-|SYNC-HAZARD|was not found/.test(fs.readFileSync(root+'/vulkan-validation.log','utf8')))throw new Error('Vulkan validation layer failure');
    await execute('/usr/local/cuda/bin/compute-sanitizer',['--tool','memcheck','--error-exitcode','1',cycle,'--check-width','64'],'resident-memcheck.log');
    await execute('/usr/local/cuda/bin/compute-sanitizer',['--tool','memcheck','--error-exitcode','1','build/neural/blitz-neural-action-gpu-tests'],'action-memcheck.log');
    const environment={BLITZ_VALIDATION_DEADLINE:String(deadline-120000),BLITZ_VALIDATION_MEMORY_MIB:'16384',BLITZ_COMPARISON_STATES:'16'};
    await execute('node',['research/neural/recovery-proof.mjs',root+'/recovery'],'recovery.log',{environment});report.recovery=read(root+'/recovery/report.json');
    await execute(cycle,[root+'/sustained','--curriculum',root+'/recovery/curriculum.json','--duration-minutes','1','--finalize-minutes','1','--states','16','--updates','128','--workers','2','--candidate-batch','4','--episode-seeds','on','--checkpoint-seconds','10','--training-profile','coverage','--quality','on','--update-backend','fused','--gpu-memory-mib','16384'],'sustained.log',{maximum:150000});report.sustained=read(root+'/sustained/report.json');
    write(root+'/diagnostic-settings.json',{...read('research/neural/refactor-smoke.json'),profile:'coverage'});
    await execute(cycle,['--audit-model','research/neural/action-diagnostic.json',root+'/sustained/initial-model.blzn',root+'/diagnostic-settings.json',root+'/sustained-initial-audit.json'],'sustained-initial-audit.log');report.sustained_initial=read(root+'/sustained-initial-audit.json');
    await execute('node',['research/neural/throughput.mjs','matrix',root+'/matrix'],'matrix.log',{environment});report.matrix=read(root+'/matrix/report.json');
    await execute('nsys',['profile','--trace=cuda,nvtx,vulkan','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--duration=15','--kill=sigterm','--output='+root+'/timeline',cycle,root+'/profile','--curriculum',root+'/matrix/curriculum.json','--states','16','--passes','8','--updates','2048','--workers','3','--candidate-batch','4','--frozen-teacher','on','--training-profile','coverage','--quality','off','--update-backend','fused','--gpu-memory-mib','16384','--minutes','2'],'timeline.log',{maximum:90000,allowFailure:true});
    await execute('nsys',['stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum','--format','csv','--output',root+'/timeline-stats',root+'/timeline.nsys-rep'],'timeline-stats.log',{maximum:60000,allowFailure:true});
    await execute('node',['research/neural/throughput.mjs','pilot',root+'/pilot'],'pilot.log',{environment,maximum:300000,allowFailure:true});report.pilot=read(root+'/pilot/report.json');
    await execute('node',['research/neural/throughput.mjs','seeds',root+'/seeds'],'seeds.log',{environment,maximum:180000,allowFailure:true});report.seeds=read(root+'/seeds/report.json');
    await execute('node',['research/neural/throughput.mjs','expanded',root+'/expanded'],'expanded.log',{environment,maximum:900000,allowFailure:true});report.expanded=read(root+'/expanded/report.json');
    const journal=read('/workspace/forensic/latest.json'),previous='/workspace/forensic/'+journal.checkpoint+'/model.blzn';
    // All validation groups, original gates; the failed training run itself is
    // never assigned an aggregate quality score.
    report.model_audit_attempts=[];
    for(const memory of [4096,16384]){
      report.previous_model=null;report.initial_model=null;
      const settings=root+'/coverage-audit-settings-'+memory+'.json';write(settings,{...read('research/neural/refactor-smoke.json'),profile:'coverage',gpu_memory_mib:memory});
      const attempt={gpu_memory_mib:memory,audits:[]};
      for(const [name,model] of [['previous',previous],['initial','/workspace/initial-model.blzn']]){
        const output=root+'/'+name+'-model-audit-'+memory+'.json';await execute(cycle,['--audit-model','research/neural/corpus-v2/validation.json',model,settings,output],name+'-model-audit-'+memory+'.log',{maximum:400000,allowFailure:true});
        if(fs.existsSync(output)){report[name+'_model']=read(output);attempt.audits.push(report[name+'_model']);}
      }
      report.model_audit_attempts.push(attempt);
      if(attempt.audits.length!==2||attempt.audits.every(a=>a.complete)||!attempt.audits.every(a=>a.rows.every(r=>r.status==='complete'||r.workspace_limited===true)))break;
    }
    if(config.optional_ablations===true&&Date.now()+180000<deadline){await execute('node',['research/neural/throughput.mjs','ablation',root+'/ablation'],'ablation.log',{environment,maximum:600000,allowFailure:true});if(fs.existsSync(root+'/ablation/report.json'))report.ablation=read(root+'/ablation/report.json');}
    if(report.ablation?.complete&&Date.now()+180000<deadline){await execute('node',['research/neural/width-quality.mjs',root+'/ablation',root+'/width-quality','128'],'width-quality.log',{environment,maximum:600000,allowFailure:true});if(fs.existsSync(root+'/width-quality/report.json'))report.width_quality=read(root+'/width-quality/report.json');}
    report.selected_width=64;report.width_quality_gate_passed=false;report.width_selection_reason='Larger widths require the three-seed validation quality gate; throughput alone cannot select them.';
    if(report.width_quality?.complete&&report.width_quality.gate.passed){report.selected_width=report.width_quality.gate.selected_width;report.width_quality_gate_passed=true;report.width_selection_reason='All three validation seeds passed unchanged audits, triangle reduction and generation time gates.';}
    report.complete=report.matrix.complete===true&&report.recovery.complete===true&&report.sustained.complete===true&&report.sustained_initial.complete===true&&report.pilot.complete===true&&report.seeds.complete===true&&report.expanded.complete===true&&report.previous_model?.complete===true&&report.initial_model?.complete===true;
  }else report.complete=true;
}catch(error){report.error=String(error);process.exitCode=1;}
finally{report.finished=Date.now();report.ready_for_next_training=report.complete&&report.cancellation?.compensated_cuda_pass===true&&report.replay?.complete===true;write(root+'/report.json',report);write('/workspace/results/job.json',{code:process.exitCode??0,experiment:'pipeline-validation',result:root+'/report.json'});}
