import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {residentArguments,preparedLearningOptions,validatePreparedLearning} from '../research/neural/resident-cycle.mjs';
import {pipelineValidationBudget,pretrainingAuthorization,actionAccrued} from '../research/neural/action-budget.mjs';
import {replayPackedDomain,soakPackedDomain} from '../research/neural/packed-domain-proof.mjs';
import {boundedProcess} from '../research/neural/bounded-process.mjs';
test('comparison runner rejects signal exits even with complete-looking child artifacts',()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-comparison-exit-')),binary=root+'/fixture.mjs';
  fs.writeFileSync(binary,`#!${process.execPath}\nimport fs from 'node:fs';const out=process.argv[2];fs.mkdirSync(out+'/data/page',{recursive:true});fs.writeFileSync(out+'/data/page/index.json',JSON.stringify({sha256:'fixed labels'}));fs.writeFileSync(out+'/report.json',JSON.stringify({complete:true,datasets:['page']}));if(process.env.BLITZ_TEST_EXIT==='signal')process.kill(process.pid,'SIGTERM');else process.exit(Number(process.env.BLITZ_TEST_EXIT));\n`,{mode:0o700});
  try{for(const status of ['0','7','signal']){
    const out=root+'/'+status,result=spawnSync(process.execPath,[fileURLToPath(new URL('../research/neural/throughput.mjs',import.meta.url)),'smoke',out],{cwd:root,env:{...process.env,BLITZ_CYCLE_BINARY:binary,BLITZ_TEST_EXIT:status,BLITZ_VALIDATION_DEADLINE:String(Date.now()+30000)},encoding:'utf8',timeout:10000});
    assert.equal(result.error,undefined);const report=JSON.parse(fs.readFileSync(out+'/report.json'));
    assert.equal(result.status,status==='0'?0:1);assert.equal(report.complete,status==='0');assert.equal(report.rows.length,status==='0'?2:1);
    if(status==='signal'){assert.equal(report.rows[0].code,null);assert.equal(report.rows[0].signal,'SIGTERM');}
  }}finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('validation subprocesses distinguish zero exit, nonzero exit, signal, timeout and cancellation',async()=>{
  const run=(source,options={})=>boundedProcess(process.execPath,['-e',source],{maximum:3000,...options});
  assert.equal((await run('process.exit(0)')).success,true);
  const failed=await run('process.exit(7)');assert.equal(failed.code,7);assert.equal(failed.success,false);
  const killed=await run("process.kill(process.pid,'SIGTERM')");assert.equal(killed.code,null);assert.equal(killed.signal,'SIGTERM');assert.equal(killed.success,false);
  const timed=await run("process.on('SIGTERM',()=>process.exit(0));setInterval(()=>{},1000)",{maximum:150,grace:100});assert.equal(timed.timed_out,true);assert.equal(timed.success,false);
  const controller=new AbortController();controller.abort();const stopped=await run('process.exit(0)',{signal:controller.signal});assert.equal(stopped.cancelled,true);assert.equal(stopped.success,false);
  const running=new AbortController(),timer=setTimeout(()=>running.abort(),100);try{const cancelled=await run('setInterval(()=>{},1000)',{signal:running.signal,grace:100});assert.equal(cancelled.cancelled,true);assert.equal(cancelled.timed_out,false);assert.equal(cancelled.success,false);}finally{clearTimeout(timer);}
});
test('packed failure replay requires accepted seeds, rejected invalid candidates and identical serial/batched outputs',async()=>{
  for(const failure of ['none','incomplete','seed','unexercised','labels','episode','signal']){
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-packed-proof-'));let calls=0;
    const ctx={root,async execute(name,args){assert.equal(name,'blitz-neural-placement-prepare');const directory=args[1];fs.mkdirSync(directory,{recursive:true});++calls;
      const index={complete:failure!=='incomplete',reference_confirmed:true,seed:{accepted:failure!=='seed'},states:16,invalid_candidates:failure==='unexercised'?0:3,sha256:failure==='labels'?String(calls):'same-labels',episode_sha256:failure==='episode'?String(calls):'same-mesh'};
      fs.writeFileSync(directory+'/index.json',JSON.stringify(index));return failure==='signal'?null:0;
    }};
    try{if(failure==='none'){assert.equal((await replayPackedDomain(ctx,1024)).complete,true);assert.equal(calls,3);}else await assert.rejects(()=>replayPackedDomain(ctx,1024),/replay/);}
    finally{fs.rmSync(root,{recursive:true,force:true});}
  }
});
test('soak gates elapsed learning, completed work and audit failures independently',async()=>{
  for(const [code,change] of [[0,{}],[null,{}],[2,{}],[0,{complete:false}],[0,{learning_elapsed_ms:1000}],[0,{failed_conditions_count:1}],[0,{coverage_quality_failed:true}],[0,{next_condition:1200}]]){
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-packed-soak-'));
    const ctx={root,async execute(name,args){assert.equal(name,'blitz-neural-cycle');fs.mkdirSync(args[0],{recursive:true});fs.writeFileSync(args[0]+'/report.json',JSON.stringify({complete:true,learning_elapsed_ms:20*60000,failed_conditions_count:0,coverage_quality_failed:false,next_condition:1700,...change}));return code;}};
    try{if(code===0&&!Object.keys(change).length)assert.equal((await soakPackedDomain(ctx)).complete,true);else await assert.rejects(()=>soakPackedDomain(ctx),/former failing condition/);}
    finally{fs.rmSync(root,{recursive:true,force:true});}
  }
});
test('validation and forensic rentals share the current pinned grant and reserve',()=>{
  const base=pretrainingAuthorization.baseline_usd;
  for(const minutes of [35,60])for(const rate of [.5,1.1,2.1]){
    const b=pipelineValidationBudget({billed:base+1,additionalAccrued:base-2.75+1,rate,minutes});
    assert.equal(b.cap_usd,base+pretrainingAuthorization.additional_usd);assert.equal(b.reserve_usd,1);assert.equal(b.maximum_rental_usd,minutes/60*(rate+.01));
    assert.ok(b.maximum_total_usd<=b.cap_usd);
  }
  for(const extra of [{billed:base+pretrainingAuthorization.additional_usd},{additionalAccrued:base+pretrainingAuthorization.additional_usd},{minutes:120},{rate:2.5},{billed:0,additionalAccrued:0}])
    assert.throws(()=>pipelineValidationBudget({billed:base,additionalAccrued:base-2.75,rate:1.1,...extra}));
  assert.equal(actionAccrued([{name:'forensic',experiment:'pipeline-validation',started_at:0,terminated_at:3600000,compute_terminated:true,deployment:{gpu_hourly_usd_cap:1.1}}]),1.11);
});
test('prepared plan reaches the real learner without inheriting old pilot settings',()=>{
  for(const workers of [1,3])for(const candidate_batch of [2,8]){
    const plan={launch:true,duration_minutes:120,finalize_minutes:7,gpu_memory_mib:8192,states_per_episode:13,updates_per_shard:64,workers,candidate_batch,hidden_width:128,checkpoint_seconds:47,update_backend:'fused',profile:'coverage',corpus:'expanded-corpus',training_selection:'expanded-selection',curriculum:'expanded-curriculum'};
    const args=residentArguments('output',{model:'verified-model',...preparedLearningOptions(plan)}),pairs=Object.fromEntries(Array.from({length:(args.length-1)/2},(_,i)=>args.slice(1+2*i,3+2*i)));
    for(const [key,value] of Object.entries({'--states':13,'--updates':64,'--workers':workers,'--candidate-batch':candidate_batch,'--hidden-width':128,'--checkpoint-seconds':47,'--corpus':'expanded-corpus','--training-selection':'expanded-selection','--curriculum':'expanded-curriculum','--episode-seeds':'on','--initialize':'verified-model','--duration-minutes':120,'--minutes':127}))assert.equal(pairs[key],String(value));
    assert.ok(!args.includes('--warmstart'));assert.throws(()=>preparedLearningOptions({...plan,launch:false}),/not authorized/);
  }
});
test('preflight rejects incomplete selected work and requires matched complete model audits',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-preflight-')),curriculum=root+'/curriculum.json';fs.writeFileSync(curriculum,JSON.stringify({conditions:[{},{}]}));
  const plan={launch:true,duration_minutes:120,finalize_minutes:10,gpu_memory_mib:8192,states_per_episode:4,updates_per_shard:32,workers:2,candidate_batch:4,hidden_width:64,checkpoint_seconds:33,update_backend:'fused',profile:'coverage',corpus:'corpus',training_selection:'training',validation_selection:'validation',curriculum};
  const previous=process.cwd();process.chdir(fileURLToPath(new URL('..',import.meta.url)));
  try{
    let incomplete=true,audits=[];
    const ctx={root,phase(){},async execute(name,args){assert.equal(name,'blitz-neural-cycle');
      if(args[0]!=='--audit-model'){assert.ok(!args.includes('--duration-minutes'));assert.equal(args[args.indexOf('--states')+1],'4');fs.mkdirSync(args[0],{recursive:true});fs.writeFileSync(args[0]+'/report.json',JSON.stringify({complete:!incomplete,failed_conditions_count:0,next_condition:2}));return incomplete?2:0;}
      const memory=JSON.parse(fs.readFileSync(args[3])).gpu_memory_mib;audits.push({memory,model:args[2]});fs.writeFileSync(args[4],JSON.stringify({complete:memory===8192,rows:[memory===4096?{status:'failed',workspace_limited:true}:{status:'complete'}]}));return 0;
    }};
    await assert.rejects(()=>validatePreparedLearning(ctx,{model:'initial'},plan),/failed preflight/);assert.equal(audits.length,0);
    incomplete=false;const proof=await validatePreparedLearning(ctx,{model:'initial'},plan);assert.ok(proof.complete);assert.deepEqual(audits.map(x=>x.memory),[4096,4096,8192,8192]);assert.deepEqual(audits.map(x=>x.model),['/workspace/previous-model.blzn','initial','/workspace/previous-model.blzn','initial']);
  }finally{process.chdir(previous);fs.rmSync(root,{recursive:true,force:true});}
});
