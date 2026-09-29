import {test} from 'node:test';
import assert from 'node:assert/strict';
import {auditProgress} from '../research/neural/audit.mjs';
import {initialization,trainerArguments,checkpointHealthy,gpuSaturated,trainingWindow,selectCalibration,trainingDeadline,shouldTrainStage} from '../research/neural/training.mjs';
test('the requested full window continues past stalled audits and the step cap',()=>{
  const config={max_steps:60,max_stalled_pilots:3};
  assert.equal(shouldTrainStage(config,20,1),true);
  assert.equal(shouldTrainStage(config,80,1),false);
  assert.equal(shouldTrainStage(config,40,3),false);
  for(const [step,stalled] of [[20,0],[80,3],[120,7]])
    assert.equal(shouldTrainStage({...config,train_until_deadline:true},step,stalled),true);
});
test('training gets its requested window after setup within the reserved absolute cutoff',()=>{
  const started=1234567;
  for(const minutes of [15,120]){
    assert.equal(trainingDeadline(started,started+180*60000,minutes),started+minutes*60000);
    assert.equal(trainingDeadline(started,started+5*60000,minutes),started+5*60000);
  }
  for(const args of [[NaN,2,1],[1,Infinity,2],[2,1,3],[1,2,0],[1,2,NaN]])
    assert.throws(()=>trainingDeadline(...args));
});
test('scratch never passes an initializer; warm start remains explicit',()=>{
  for(const batch of [3,8]){
    const config={batch,core:128,checkpoint_every:7,workers:4,gpu_memory_mib:2048};
    const scratch=trainerArguments('data','run',initialization('--from-scratch'),config,100,2);
    assert.ok(!scratch.includes('--initialize'));
    const warm=trainerArguments('data','run',initialization('saved.blzn'),config,100,2);
    assert.deepEqual(warm.slice(-2),['--initialize','saved.blzn']);
    assert.equal(scratch[scratch.indexOf('--workers')+1],'4');
    assert.equal(scratch[scratch.indexOf('--gpu-memory-mib')+1],'2048');
  }
});
const health={finite:true,restored:true,optimizer_restored:true,native_max_abs_error:1e-5,parameter_change:.02,gradient_norm:.3};
test('calibration selects useful throughput among restored models with sustained GPU evidence',()=>{
  const gpu={samples:70,seconds:69,mean:96,p10:92};
  const trial={health,gpu,measured_steps:20,vertices_per_second:100};
  const fast={...trial,vertices_per_second:200};
  assert.equal(selectCalibration([trial,fast,{...trial,vertices_per_second:500,health:{...health,optimizer_restored:false}}]),fast);
  for(const h of [{...health,native_max_abs_error:NaN},{...health,gradient_norm:0},{...health,parameter_change:0}])assert.equal(checkpointHealthy(h),false);
  assert.throws(()=>selectCalibration([{...trial,measured_steps:0}]),/No calibration/);
  for(const insufficient of [undefined,{...gpu,samples:20},{...gpu,seconds:30},{...gpu,mean:83},{...gpu,p10:74}]){
    assert.equal(gpuSaturated(insufficient),false);
    assert.equal(selectCalibration([trial,{...fast,gpu:insufficient}]),trial);
    assert.throws(()=>selectCalibration([{...fast,gpu:insufficient}]),/No calibration/);
  }
});
test('utilization evidence needs measured time and excludes audit activity',()=>{
  const samples=Array.from({length:61},(_,i)=>({at:new Date(i*1000).toISOString(),phase:'training',gpu:95,power_w:300,memory_mib:1000}));
  assert.equal(trainingWindow(samples).seconds,60);
  assert.equal(trainingWindow(samples).mean,95);
  samples.push({at:new Date(61000).toISOString(),phase:'audit',gpu:0});
  assert.equal(trainingWindow(samples).samples,61);
  assert.equal(trainingWindow(samples.slice(0,20)).seconds,19);
  samples.push({at:new Date(90000).toISOString(),phase:'training',gpu:99,power_w:310,memory_mib:1100});
  assert.equal(trainingWindow(samples).samples,1); // Audit gaps cannot pad the sustained window.
});
test('resource limits identify the asset immediately and retain the unscored summary',()=>{
  for(const [id,kind] of [['tree','sample_count'],['rock','workspace_memory']]){
    const summary={complete:false,completed:3,expected:8,score:null,blocked_assets:[{id,diagnostic:{kind,requested:100,limit:64}}]};
    assert.throws(()=>auditProgress(summary),error=>error.audit===summary&&error.message.includes(id)&&error.message.includes(kind));
    assert.equal(summary.score,null);
  }
});
test('deadline progress resumes; stagnation stops; a complete audit finishes',()=>{
  assert.equal(auditProgress({complete:false,completed:2},1),2);
  assert.equal(auditProgress({complete:false,completed:0}),0);
  assert.throws(()=>auditProgress({complete:false,completed:2},2),/no progress/);
  assert.equal(auditProgress({complete:true,completed:8,score:0},8),8);
});
test('a processed batch containing bake exceptions cannot pass readiness',()=>{
  const summary={complete:true,completed:8,score:0,failed_assets:[{id:'mesh',failure:'invalid output'}]};
  assert.throws(()=>auditProgress(summary),error=>error.audit===summary&&error.message.includes('invalid output'));
});
