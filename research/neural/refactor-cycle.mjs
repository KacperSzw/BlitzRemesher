// Frozen first v3 learning schedule. Development diagnostics never enter training
// and this bounded screening does not produce a protocol SCORE or release claim.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {read,write} from './runpod-api.mjs';
import {actionHealth,auditRowsHealthy} from './action-gates.mjs';
export const cycleConditions=Object.freeze([
  {asset:'ph_painted_wooden_bench',pixels:32,previous:0},
  {asset:'ph_sweet_potato',pixels:64,previous:0},
  {asset:'ph_namaqualand_boulder_04',pixels:128,previous:0},
  {asset:'ph_painted_wooden_bench',pixels:128,previous:2},
  {asset:'ph_sweet_potato',pixels:128,previous:2},
  {asset:'ph_namaqualand_boulder_04',pixels:64,previous:2}
]);
export const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
export function modelPayload(file){const b=fs.readFileSync(file),start=20+b.readUInt32LE(16),end=start+4*b.readUInt32LE(12);if(b.toString('ascii',0,8)!=='BLZNET01'||end+64!==b.length)throw new Error('Invalid model envelope');return hashBytes(b.subarray(start,end));}
const hashBytes=b=>crypto.createHash('sha256').update(b).digest('hex');
export function trainingHealth(directory,step){const h=read(directory+'/latest.json');if(!actionHealth(h)||h.step!==step||hash(directory+'/'+h.model)!==h.model_sha256||hash(directory+'/'+h.checkpoint)!==h.checkpoint_sha256)throw new Error('Optimizer, checkpoint or export health failed');return h;}
export function datasetHealth(directory,previous=false){const j=read(directory+'/index.json');return j.schema===3&&j.complete===true&&j.reference_confirmed===true&&j.states>0&&(!previous||j.preceding_lod_emitted&&j.previous_triangles<j.source_triangles)&&hash(directory+'/actions.bin')===j.sha256&&hash(directory+'/contract.json')===j.contract_sha256;}
export function cycleWindow(start,latest,minutes=120){if(!Number.isFinite(start)||!Number.isFinite(latest)||minutes!==120||latest-start<minutes*60000+15000)throw new Error('A full two-hour learning cycle no longer fits');return {started:start,deadline:start+minutes*60000,minutes};}
async function audit({root,execute,phase},model,name,confirmation='gpu'){
  phase('audit');const runs=[],manifest='research/neural/action-diagnostic.json',config='research/neural/refactor-smoke.json',assets=read(manifest).assets;
  for(const ranking of ['constant','learned']){const out=root+'/'+name+'/'+ranking;
    const code=await execute('blitz',['bench',manifest,config,out,'--neural-model',model,'--neural-control',ranking,'--ranking-seed','101','--action-trials','8','--action-batch','16','--gpu-memory-mib','16384','--neural-confirmation',confirmation,'--minutes','3'],root+'/'+name+'-'+ranking+'.log',3.2);
    const summary=read(out+'/summary.json'),rows=assets.map(a=>out+'/rows/'+a.id+'.json').filter(fs.existsSync).map(read);
    const healthy=code===0&&summary.complete&&auditRowsHealthy(rows,assets);runs.push({ranking,healthy,summary,rows});
    write(root+'/'+name+'/report.json',{complete:false,score:null,release_quality_proven:false,runs,model_sha256:hash(model),config_sha256:hash(config)});
    if(!healthy)throw new Error('Incomplete or unhealthy development audit: '+name+'/'+ranking);
  }
  const report={complete:true,score:null,release_quality_proven:false,runs,model_sha256:hash(model),config_sha256:hash(config)};write(root+'/'+name+'/report.json',report);return report;
}
export async function validateRefactor(ctx){const {root,execute,phase}=ctx;phase('validation');
  await execute('blitz-neural-profile',[],root+'/profile.json',1);const profile=read(root+'/profile.json');
  if(!profile.topology.every(r=>r.exact_output)||!profile.audit.every(r=>r.exact_metrics))throw new Error('GPU profile parity failed');
  await execute('blitz-neural-action-train',['--replay','research/neural/evidence/gpu-refactor-local/replay',root+'/cross-gpu.json'],root+'/cross-gpu.log',1);
  if(!read(root+'/cross-gpu.json').passed)throw new Error('Cross-GPU FP32/FP64 export parity failed');
  const dataset='research/neural/evidence/action-screening-v3/action-v2-staged/curriculum',bench=[];
  for(let repeat=0;repeat<3;repeat++)for(const backend of repeat%2?['eager','captured']:['captured','eager']){
    const run=root+`/benchmark-${backend}-${repeat}`;await execute('blitz-neural-action-train',[dataset,run,'--steps','1024','--batch','512','--seed','101','--minutes','1','--backend',backend],run+'.log',1.1);
    const health=trainingHealth(run,1024);bench.push({backend,repeat,health,payload:modelPayload(run+'/'+health.model)});
  }
  if(new Set(bench.map(b=>b.payload)).size!==1)throw new Error('Eager/captured update payloads differ');
  const data=root+'/data',shard=data+'/validation';fs.mkdirSync(data,{recursive:true});
  await execute('blitz-neural-placement-prepare',['ph_painted_wooden_bench',shard,'--states','2','--pool','2','--pixels','32','--previous-steps','1','--gpu-memory-mib','16384','--minutes','1'],root+'/teacher-validation.log',1.1);
  if(!datasetHealth(shard,true))throw new Error('Placement teacher preflight failed');
  write(data+'/index.json',{datasets:['validation']});
  const resume=root+'/validation-resume',whole=root+'/validation-whole';
  for(const [run,steps]of [[resume,512],[resume,1024],[whole,1024]])await execute('blitz-neural-action-train',[data,run,'--steps',String(steps),'--batch','512','--seed','101','--minutes','1'],run+'.log',1.1);
  const health=trainingHealth(resume,1024),reference=trainingHealth(whole,1024);
  if(modelPayload(resume+'/'+health.model)!==modelPayload(whole+'/'+reference.model))throw new Error('Split-run continuation differs');
  // New curriculum contracts retain optimizer state; same-data continuation must
  // remain bit-exact before using this path for refreshed examples.
  const continuation=root+'/validation-warmstart';await execute('blitz-neural-action-train',[data,continuation,'--warmstart',resume+'/'+health.checkpoint,'--steps','1536','--batch','512','--seed','101','--minutes','1'],continuation+'.log',1.1);
  await execute('blitz-neural-action-train',[data,whole,'--steps','1536','--batch','512','--seed','101','--minutes','1'],whole+'.log',1.1);
  const carried=trainingHealth(continuation,1536),continued=trainingHealth(whole,1536);
  if(modelPayload(continuation+'/'+carried.model)!==modelPayload(whole+'/'+continued.model))throw new Error('Optimizer carry-forward differs');
  const quality=await audit(ctx,continuation+'/'+carried.model,'preflight-audit');
  const result={complete:true,profile,bench,teacher:read(shard+'/index.json'),quality,exact_continuation:true,exact_curriculum_continuation:true,cross_gpu:read(root+'/cross-gpu.json'),model:continuation+'/'+carried.model,checkpoint:continuation+'/'+carried.checkpoint,step:carried.step};
  write(root+'/validation.json',result);return result;
}
export function architectureReport(validation){const median=a=>a.sort((a,b)=>a-b)[Math.floor(a.length/2)],bench=validation.bench;
  const captured=median(bench.filter(r=>r.backend==='captured').map(r=>r.health.update_seconds)),eager=median(bench.filter(r=>r.backend==='eager').map(r=>r.health.update_seconds));
  return `# GPU learning pipeline before the two-hour cycle\n\n\`\`\`mermaid\nflowchart LR\n IO["CPU: checked asset and dataset I/O"] --> M["GPU: resident mesh + topology"]\n M --> P["GPU: 128 → 64 → 64 → 12 policy"]\n P --> G["GPU: XYZ + separate wedge normals"]\n G --> A["GPU: source + previous LOD audits"]\n A --> D["Packed queried examples"]\n D --> U["Captured sampler → MLP → loss → AdamW"]\n U --> P\n A --> E["CPU: checkpoint/export + schedule"]\n\`\`\`\n\nGPU: ${validation.profile.gpu}. Median 1,024-update device window: captured ${captured.toFixed(6)} s, eager ${eager.toFixed(6)} s (${(eager/captured).toFixed(2)}×). Startup, capture and checkpoint times are separate in validation.json. Eager/captured and interrupted/continued model payloads agree exactly. Cross-GPU native/FP64 replay passed.\n\nTopology and cache timings, raster counts, transfers and useful accepted actions are in profile.json. Audits still orchestrate views/refinement and tile allocation on the host; all raster, distance and appearance calculations remain on GPU. Audit FP64 work, repeated candidate renders and teacher search dominate the learning cycle; optimizer throughput is not a quality score.\n\nV3: 13,196 FP32 parameters (52,784 bytes). Each action keeps 86 FP32 values, 32 flag bits, 8 label bits and up to 9 masked FP32 targets; 8 conditions are shared by a state. Geometry IDs are uint32; material IDs are uint16. No lossy feature quantization.\n\nThe frozen two-asset diagnostic uses fewer views than the full protocol. Audited cameras do not guarantee every view or global optimality. Hard source/previous gates are independent of learned pass logits. Missing normal streams remain flat shaded.\n`;
}
export async function learningCycle(ctx,validation,{now=Date.now,minutes=120}={}){
  const {root,execute,phase,latest}=ctx,window=cycleWindow(now(),latest,minutes),data=root+'/data';
  const result={...window,complete:false,score:null,release_quality_proven:false,training_steps:0,datasets:['validation'],stages:[],audits:[],conditions:cycleConditions};
  let model=validation.model,checkpoint=validation.checkpoint,step=validation.step,nextAudit=window.started+30*60000,iteration=0;
  const states=cycleConditions.map(()=>8);
  const finalAuditBudget=Math.min(6.5*60000,Math.max(30000,validation.quality.runs.reduce((n,r)=>n+r.summary.seconds,0)*2000+15000));
  write(root+'/cycle.json',result);
  while(now()<window.deadline-15000){
    if(window.deadline-now()<=finalAuditBudget){const report=await audit(ctx,model,'final-audit');result.audits.push({at:now(),step,report,final:true});result.final_audited_model=model;break;}
    if(now()>=nextAudit&&window.deadline-now()>7*60000){const report=await audit(ctx,model,'audit-'+result.audits.length);result.audits.push({at:now(),step,report});nextAudit+=30*60000;write(root+'/cycle.json',result);continue;}
    const remaining=window.deadline-now();
    // Near the end, spend remaining time on verified updates, not a shard that
    // cannot finish. Every checkpoint remains numerically checked and resumable.
    if(remaining<90000){phase('training');const out=root+'/tail-'+iteration++,target=step+4096;
      const code=await execute('blitz-neural-action-train',[data,out,'--warmstart',checkpoint,'--steps',String(target),'--batch','512','--seed','101','--minutes',String(Math.max(.05,(remaining-15000)/60000))],out+'.log',remaining/60000);
      const h=read(out+'/latest.json');if(!actionHealth(h)||h.step<=step)throw new Error('Final optimizer segment unhealthy');result.training_steps+=h.step-step;step=h.step;checkpoint=out+'/'+h.checkpoint;model=out+'/'+h.model;result.stages.push({phase:'training',code,health:h,at:now()});write(root+'/cycle.json',result);continue;}
    const index=iteration%cycleConditions.length,c=cycleConditions[index],name='shard-'+iteration++,out=data+'/'+name,started=now();phase('preparation');
    const budget=Math.min(3,(remaining-45000)/60000),code=await execute('blitz-neural-placement-prepare',[c.asset,out,'--states',String(states[index]),'--pool','4','--pixels',String(c.pixels),'--previous-steps',String(c.previous),'--seed',String(101+iteration),'--model',model,'--gpu-memory-mib','16384','--minutes',String(budget)],root+'/'+name+'.log',budget+.15);
    const usable=code===0&&datasetHealth(out,!!c.previous),j=read(out+'/index.json');result.stages.push({phase:'preparation',condition:c,code,usable,index:j,at:now()});
    if(!usable){states[index]=Math.max(1,Math.floor(states[index]/2));write(root+'/cycle.json',result);continue;}
    if(now()-started<45000)states[index]=Math.min(64,states[index]*2);
    result.datasets.push(name);write(data+'/index.json',{datasets:result.datasets});write(data+'/index-'+iteration+'.json',{datasets:result.datasets});
    phase('training');const train=root+'/train-'+iteration,target=step+2048;
    await execute('blitz-neural-action-train',[data,train,'--warmstart',checkpoint,'--steps',String(target),'--batch','512','--seed','101','--minutes','1'],train+'.log',1.1);
    const h=trainingHealth(train,target);result.training_steps+=h.step-step;step=h.step;checkpoint=train+'/'+h.checkpoint;model=train+'/'+h.model;result.stages.push({phase:'training',health:h,at:now()});
    Object.assign(result,{model,checkpoint,step});write(root+'/cycle.json',result);
  }
  Object.assign(result,{complete:result.datasets.length>1&&result.training_steps>0&&result.audits.length>=3&&result.final_audited_model===model,finished:now(),model,checkpoint,step,active_seconds:(now()-window.started)/1000});write(root+'/cycle.json',result);return result;
}
