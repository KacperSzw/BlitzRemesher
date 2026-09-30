import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {write,read} from '../research/neural/runpod-api.mjs';
import {refactorBudget,refactorAuthorization,actionAccrued,hardwareValidationBudget} from '../research/neural/action-budget.mjs';
import {cycleWindow,cycleConditions,modelPayload,hash,learningCycle} from '../research/neural/refactor-cycle.mjs';
test('hardware validation consumes the existing grant and includes previous validation rentals',()=>{
  const prior=refactorAuthorization.baseline_usd+5.3;const b=hardwareValidationBudget({billed:prior,additionalAccrued:prior-2.75,rate:2.5});assert.ok(b.maximum_total_usd<=b.cap_usd);assert.equal(b.minutes,35);
  for(const extra of [{billed:prior+1},{minutes:120},{rate:3},{additionalAccrued:prior}])assert.throws(()=>hardwareValidationBudget({billed:prior,additionalAccrued:prior-2.75,rate:2.5,...extra}));
  assert.equal(actionAccrued([{name:'hardware-test',experiment:'hardware-validation',started_at:0,terminated_at:3600000,compute_terminated:true,deployment:{gpu_hourly_usd_cap:2.5}}]),2.51);
});
test('new authorization includes historical ledger, future rentals, rate, storage and reserve',()=>{
  const base=refactorAuthorization.baseline_usd;
  for(const rate of [1.9,2.09,2.5]){const b=refactorBudget({billed:9.49,rate,additionalAccrued:base-2.75});assert.ok(b.maximum_additional_usd<=8);assert.equal(b.maximum_rental_usd,160/60*(rate+.01));assert.equal(b.cap_usd,base+8);}
  for(const input of [{billed:base+2,rate:2.5,additionalAccrued:base-2.75},{billed:9,rate:2,additionalAccrued:0},{billed:9,rate:3,additionalAccrued:base-2.75},{billed:9,rate:2,additionalAccrued:base-2.75,minutes:120}])assert.throws(()=>refactorBudget(input));
  assert.equal(actionAccrued([{name:'refactor',experiment:'gpu-refactor',started_at:0,terminated_at:3600000,compute_terminated:true,deployment:{gpu_hourly_usd_cap:2.5}}]),2.51);
});
test('a complete learning window cannot be silently shortened after validation',()=>{
  for(const start of [0,99999]){assert.equal(cycleWindow(start,start+121*60000).deadline,start+120*60000);assert.throws(()=>cycleWindow(start,start+119*60000));assert.throws(()=>cycleWindow(start,start+130*60000,90));}
});
test('refresh identities remain training-only and include preceding LOD and 128px conditions',()=>{
  const allowed=new Set(read('research/neural/training-manifest.json').assets.map(a=>a.id)),development=new Set(read('research/pilot.json').assets.map(a=>a.id));
  assert.ok(cycleConditions.every(c=>allowed.has(c.asset)&&!development.has(c.asset)));assert.ok(cycleConditions.some(c=>c.previous>0));assert.ok(cycleConditions.some(c=>c.pixels===128));
});
function model(file,provenance,values){const b=Buffer.alloc(20+provenance.length+values.length*4+64);b.write('BLZNET01');b.writeUInt32LE(3,8);b.writeUInt32LE(values.length,12);b.writeUInt32LE(provenance.length,16);b.write(provenance,20);values.forEach((v,i)=>b.writeFloatLE(v,20+provenance.length+i*4));b.fill(provenance.charCodeAt(0),b.length-64);fs.writeFileSync(file,b);}
test('model comparison includes all weights but excludes metadata and its checksum',()=>{
  const d=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-payload-'));try{model(d+'/a','a',[1,2]);model(d+'/b','long',[1,2]);model(d+'/c','a',[1,3]);assert.equal(modelPayload(d+'/a'),modelPayload(d+'/b'));assert.notEqual(modelPayload(d+'/a'),modelPayload(d+'/c'));}finally{fs.rmSync(d,{recursive:true,force:true});}
});
test('learning cycle excludes incomplete labels, carries optimizer state and audits the final model',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-cycle-'));let time=0,preparations=0,updates=0;
  try{fs.mkdirSync(root+'/data');write(root+'/data/index.json',{datasets:['validation']});fs.writeFileSync(root+'/initial.pt','initial');model(root+'/initial.blzn','init',[1]);
    const execute=async(name,args)=>{
      if(name==='blitz-neural-placement-prepare'){preparations++;time+=2*60000;const out=args[1];fs.mkdirSync(out,{recursive:true});fs.writeFileSync(out+'/actions.bin','labels');write(out+'/contract.json',{teacher:true});write(out+'/index.json',{schema:3,complete:preparations>1,states:2,reference_confirmed:true,preceding_lod_emitted:true,source_triangles:100,previous_triangles:90,sha256:hash(out+'/actions.bin'),contract_sha256:hash(out+'/contract.json')});return preparations===1?2:0;}
      if(name==='blitz-neural-action-train'){updates++;time+=60000;assert.ok(fs.existsSync(args[args.indexOf('--warmstart')+1]));assert.ok(!read(root+'/data/index.json').datasets.includes('shard-0'));const out=args[1],step=Number(args[args.indexOf('--steps')+1]);fs.mkdirSync(out,{recursive:true});model(out+'/model.blzn','step'+step,[step]);fs.writeFileSync(out+'/checkpoint.pt',String(step));write(out+'/latest.json',{step,complete:true,finite:true,restored:true,optimizer_restored:true,native_max_abs:0,fp64_max_abs:0,first_loss:1,last_loss:.1,gradient_norm:.1,parameter_change:.1,preferred_membership:.5,model:'model.blzn',checkpoint:'checkpoint.pt',model_sha256:hash(out+'/model.blzn'),checkpoint_sha256:hash(out+'/checkpoint.pt')});return 0;}
      assert.equal(name,'blitz');time+=30000;const out=args[3];write(out+'/summary.json',{complete:true,seconds:30,score:null});for(const a of read(args[1]).assets)write(out+'/rows/'+a.id+'.json',{id:a.id,complete:true,neural:{}});return 0;
    };
    const r=await learningCycle({root,execute,phase:()=>{},latest:130*60000},{model:root+'/initial.blzn',checkpoint:root+'/initial.pt',step:0,quality:{runs:[{summary:{seconds:30}},{summary:{seconds:30}}]}},{now:()=>time});
    assert.equal(r.complete,true);assert.ok(preparations>1&&updates>1&&r.training_steps>0);assert.equal(r.model,r.final_audited_model);assert.ok(r.audits.length>=3);assert.ok(r.finished<=r.deadline);assert.ok(r.active_seconds>=117*60);assert.equal(r.score,null);
  }finally{fs.rmSync(root,{recursive:true,force:true});}
});
