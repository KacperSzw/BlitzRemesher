import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {generalize} from '../research/neural/action-generalize.mjs';
import {write} from '../research/neural/runpod-api.mjs';
import {actionBudget,actionAccrued} from '../research/neural/action-budget.mjs';
import {endpointDecision,pilotDecision,actionHealth} from '../research/neural/action-gates.mjs';
test('cumulative rental cap includes previous spend, storage allowance and reserve',()=>{
  for(const rate of [1.8,2.1,2.5])for(const billed of [1.9,2.75,4.2]){
    const b=actionBudget({rate,billed,minutes:60});assert.ok(b.maximum_total_usd<=10);assert.ok(b.prior_assumed_usd>=billed);assert.ok(b.reserve_usd>=1);
    assert.equal(b.maximum_rental_usd,rate+.01);
  }
  for(const invalid of [{billed:9,rate:2},{billed:2,rate:5,minutes:150},{billed:NaN,rate:2},{billed:2,rate:2,minutes:Infinity},{billed:2,rate:2,reserve:0}])assert.throws(()=>actionBudget(invalid));
  const stopped={name:'one',experiment:'action-v2',started_at:1000,terminated_at:1801000,compute_terminated:true};
  const accrued=actionAccrued([stopped,stopped,{name:'old',experiment:'vertex-v1'}],3601000,2);
  assert.equal(accrued,1);assert.equal(actionBudget({billed:2,rate:2,priorEstimate:3,additionalAccrued:accrued}).prior_assumed_usd,4);
  assert.equal(actionBudget({billed:5,rate:2,priorEstimate:3,additionalAccrued:accrued}).prior_assumed_usd,5);
  assert.throws(()=>actionBudget({billed:2,rate:2,additionalAccrued:6}));
  const pilot={...stopped,name:'pilot',experiment:'action-v2-pilot',compute_terminated:false};
  assert.equal(actionAccrued([stopped,pilot,pilot],3601000,2),3);
  assert.throws(()=>actionBudget({billed:2,rate:2.5,minutes:90,additionalAccrued:3}));
});
test('action health rejects unverified updates and unchanged or nonfinite parameters',()=>{
  const health={complete:true,finite:true,restored:true,optimizer_restored:true,native_max_abs:1e-5,fp64_max_abs:2e-5,first_loss:1,last_loss:.2,gradient_norm:.3,parameter_change:.1,preferred_membership:.9};
  assert.equal(actionHealth(health),true);
  for(const field of ['complete','finite','restored','optimizer_restored'])assert.equal(actionHealth({...health,[field]:false}),false);
  for(const field of ['native_max_abs','fp64_max_abs','first_loss','last_loss','gradient_norm','parameter_change','preferred_membership'])assert.equal(actionHealth({...health,[field]:NaN}),false);
  for(const field of ['native_max_abs','fp64_max_abs'])assert.equal(actionHealth({...health,[field]:.000201}),false);
  assert.equal(actionHealth({...health,parameter_change:0}),false);
  assert.equal(actionHealth({...health,preferred_membership:1.01}),false);
});
test('cloud continuation stops before training when the preceding LOD is unverified',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-action-'));let calls=0;
  try{await assert.rejects(generalize({root,deadline:Date.now()+120000,phase:()=>{},execute:async(name,args)=>{
    ++calls;assert.equal(name,'blitz-neural-action-prepare');write(args[1]+'/index.json',{complete:true,reference_confirmed:false});
  }}),/audited preceding LOD/);assert.equal(calls,1);}finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('an incomplete matched pilot prevents a second cloud training stage',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-action-'));let updates=0,audits=0;
  try{const result=await generalize({root,deadline:Date.now()+120000,phase:()=>{},execute:async(name,args)=>{
    if(name==='blitz-neural-action-prepare')write(args[1]+'/index.json',{complete:true,reference_confirmed:true,preceding_lod_emitted:true,source_triangles:100,previous_triangles:90});
    else if(name==='blitz-neural-action-train'){++updates;write(args[1]+'/latest.json',{complete:true,finite:true,restored:true,optimizer_restored:true,native_max_abs:1e-5,fp64_max_abs:2e-5,first_loss:1,last_loss:.2,gradient_norm:.3,parameter_change:.1,preferred_membership:.9,model:'model.blzn'});}
    else{assert.equal(name,'node');++audits;write(args[1]+'/report.json',{complete:false,gate:{passed:false,reason:'incomplete'},persisted:false});}
  }});assert.equal(updates,3);assert.equal(audits,1);assert.equal(result.complete,false);assert.equal(result.generalization_gate_passed,false);assert.equal(result.stop_reason,'incomplete_matched_pilot');
  }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('matched pilot requires three seeds, all controls, complete assets and measured advantage',()=>{
  const methods=['learned','learned','learned','constant','shuffled','shuffled','shuffled','shortest','current-plane'];
  const runs=methods.map(ranking=>({ranking,code:0,summary:{complete:true,completed:8,expected:8,score:ranking==='learned'?8:5,categories:{a:{count:2,mean_ratio:.8},b:{count:2,mean_ratio:.9}}}}));
  for(const scoreGain of [1,2])assert.equal(pilotDecision(runs,{scoreGain}).passed,true);
  assert.equal(pilotDecision(runs,{scoreGain:4}).passed,false);
  runs[1].summary.complete=false;assert.equal(pilotDecision(runs).reason,'incomplete');runs[1].summary.complete=true;
  runs[6].summary.score=8;assert.equal(pilotDecision(runs).passed,false);
});
const row=(membership,reduction,passed=false)=>({health:{finite:true,restored:true,optimizer_restored:true},proof:{complete:true,preferred_membership:membership,proof_passed:passed,rows:[{reduction}]}});
test('endpoint evidence must persist, while two flat audits stop further spending',()=>{
  assert.equal(endpointDecision([row(.5,10),row(.7,10)]).stop,false);
  assert.equal(endpointDecision([row(.5,10),row(.5,10),row(.5,10)]).reason,'two_flat_action_audits');
  assert.equal(endpointDecision([row(.96,10,true)]).passed,false);
  assert.equal(endpointDecision([row(.96,10,true),row(.97,10,true)]).passed,true);
  const bad=row(.99,10,true);bad.health.finite=false;assert.equal(endpointDecision([bad]).reason,'health_or_incomplete_audit');
  const partial=row(.99,10,true);partial.proof.complete=false;assert.equal(endpointDecision([partial]).passed,false);
});
