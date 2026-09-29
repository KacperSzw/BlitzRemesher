import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {generalize} from '../research/neural/action-generalize.mjs';
import {staged} from '../research/neural/action-staged.mjs';
import {freshConditions} from '../research/neural/action-curriculum.mjs';
import {evaluateSaved,fullPilotFits} from '../research/neural/action-evaluate.mjs';
import {write} from '../research/neural/runpod-api.mjs';
import {actionBudget,actionAccrued,continuationBudget,continuationAuthorization} from '../research/neural/action-budget.mjs';
import {endpointDecision,pilotDecision,actionHealth,comparisonMethods,auditRowsHealthy} from '../research/neural/action-gates.mjs';
const trainedHealth=args=>({complete:true,finite:true,restored:true,optimizer_restored:true,native_max_abs:1e-5,fp64_max_abs:2e-5,first_loss:1,last_loss:.2,gradient_norm:.3,parameter_change:.1,preferred_membership:.9,model:'model.blzn',step:Number(args[args.indexOf('--steps')+1]),steps_this_segment:8192});
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
  const evaluation={...stopped,name:'evaluation',experiment:'action-v2-evaluate'};
  assert.equal(actionAccrued([stopped,pilot,evaluation,evaluation],3601000,2),4);
  assert.throws(()=>actionBudget({billed:2,rate:2.5,minutes:90,additionalAccrued:3}));
});
test('additional authorization counts earlier rentals and retries without adding to the old ceiling',()=>{
  const base=continuationAuthorization.baseline_usd,accrued=base-2.75;
  for(const rate of [1.9,2.5])for(const minutes of [60,120,150]){
    const b=continuationBudget({billed:4,rate,minutes,additionalAccrued:accrued});
    assert.ok(b.maximum_additional_usd<=8);assert.equal(b.cap_usd,base+8);
    assert.equal(b.prior_assumed_usd,base);
  }
  const retry=continuationBudget({billed:4,rate:2,minutes:60,additionalAccrued:accrued+1});
  assert.equal(retry.prior_assumed_usd,base+1);
  assert.throws(()=>continuationBudget({billed:4,rate:2,additionalAccrued:0}),/missing earlier/);
  assert.throws(()=>continuationBudget({billed:base+7,rate:2,additionalAccrued:accrued}),/cap/);
  assert.throws(()=>continuationBudget({billed:4,rate:2,additionalAccrued:accrued+7}),/cap/);
  assert.equal(actionAccrued([{name:'new',experiment:'action-v2-staged',started_at:0,terminated_at:1800000,compute_terminated:true}],0,2),1);
  const rental={experiment:'action-v2-staged',started_at:0,terminated_at:3600000,compute_terminated:true};
  const mixed=actionAccrued([{...rental,name:'one',deployment:{gpu_hourly_usd_cap:2}},{...rental,name:'two',deployment:{gpu_hourly_usd_cap:3}}]);
  assert.equal(mixed,5.02);
  assert.throws(()=>actionAccrued([{...rental,name:'bad',deployment:{gpu_hourly_usd_cap:NaN}}]),/rate/);
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
test('saved-model evaluation never trains and stops after an incomplete diagnostic',async()=>{
  for(const complete of [false,true]){
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-evaluate-'));const calls=[];
    try{const result=await evaluateSaved({root,deadline:Date.now()+120000,phase:()=>{},execute:async(name,args)=>{
      calls.push(name);assert.equal(name,'node');assert.equal(args[0],'research/neural/action-pilot.mjs');assert.equal(args[args.indexOf('--neural-confirmation')+1],'gpu');
      write(args[1]+'/report.json',{complete,runs:Array.from({length:9},()=>({wall_seconds:30}))});
    }});assert.deepEqual(calls,['node']);assert.equal(result.training_steps,0);assert.equal(result.complete,complete);assert.equal(result.quality_proven,false);
    }finally{fs.rmSync(root,{recursive:true,force:true});}
  }
  assert.equal(fullPilotFits({complete:true,runs:Array.from({length:9},()=>({wall_seconds:2}))},300000),true);
  assert.equal(fullPilotFits({complete:false,runs:[]},300000),false);
  assert.equal(fullPilotFits({complete:true,runs:Array.from({length:9},()=>({wall_seconds:NaN}))},300000),false);
});
test('cloud continuation stops before training when the preceding LOD is unverified',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-action-'));let calls=0;
  try{await assert.rejects(generalize({root,deadline:Date.now()+120000,phase:()=>{},reusePrepared:false,execute:async(name,args)=>{
    ++calls;assert.equal(name,'blitz-neural-action-prepare');write(args[1]+'/index.json',{complete:true,reference_confirmed:false});
  }}),/audited preceding LOD/);assert.equal(calls,1);}finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('staged continuation refuses training after missing, cancelled or unhealthy baseline rows',async()=>{
  for(const failure of ['missing','cancelled','resource','nonfinite','disagreement']){
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-staged-'));let calls=0;
    try{const result=await staged({root,deadline:Date.now()+4800000,phase:()=>{},execute:async(name,args)=>{
      ++calls;assert.equal(name,'blitz');assert.equal(args[args.indexOf('--neural-confirmation')+1],'gpu');
      const out=args[3],assets=JSON.parse(fs.readFileSync(args[1])).assets;
      write(out+'/summary.json',{complete:true,seconds:10});
      for(const a of failure==='missing'?assets.slice(0,1):assets)write(out+'/rows/'+a.id+'.json',{id:a.id,complete:failure!=='cancelled',neural:{resource_failures:Number(failure==='resource'),confirmation_nonfinite:Number(failure==='nonfinite'),confirmation_disagreements:Number(failure==='disagreement')}});
    }});assert.equal(calls,1);assert.equal(result.training_steps,0);assert.equal(result.complete,false);
    }finally{fs.rmSync(root,{recursive:true,force:true});}
  }
});
test('fresh curriculum is confined to frozen training identities and covers a third category',()=>{
  const corpus=JSON.parse(fs.readFileSync(new URL('../research/corpus.json',import.meta.url))).assets,allowed=new Set(JSON.parse(fs.readFileSync(new URL('../research/neural/training-manifest.json',import.meta.url))).assets.map(a=>a.id));
  const categories=new Set();for(const c of freshConditions){assert.ok(allowed.has(c.asset));categories.add(corpus.find(a=>a.id===c.asset).category);}
  assert.equal(categories.size,3);
});
test('screening fixes one seed and one control without satisfying the full pilot',()=>{
  const models=['one','two','three'],screen=comparisonMethods(models,'screening');
  assert.deepEqual(screen.map(m=>m.name),['constant','learned-0']);
  assert.ok(screen.every(m=>m.model===models[0]&&m.seed===1));
  assert.equal(pilotDecision(screen).passed,false);
  for(const scenario of ['pilot','diagnostic']){
    const methods=comparisonMethods(models,scenario);
    assert.equal(methods.length,9);assert.equal(methods.filter(m=>m.ranking==='learned').length,3);
    assert.equal(methods.filter(m=>m.ranking==='shuffled').length,3);
  }
  assert.throws(()=>comparisonMethods(models,'unknown'));
  assert.throws(()=>comparisonMethods(models.slice(1),'screening'));
});
test('audit health requires every identity, complete rows and clean numerical/resource verdicts',()=>{
  const assets=[{id:'a'},{id:'b'}],rows=assets.map(a=>({...a,complete:true,neural:{}}));
  assert.equal(auditRowsHealthy(rows,assets),true);
  for(const bad of [[],rows.slice(1),[rows[0],rows[0]],[rows[0],{...rows[1],id:'c'}],
    [rows[0],{...rows[1],complete:false}],[rows[0],{...rows[1],failed:true}],
    [rows[0],{...rows[1],neural:undefined}]])assert.equal(auditRowsHealthy(bad,assets),false);
  for(const field of ['resource_failures','confirmation_resources','confirmation_cancelled','confirmation_nonfinite','confirmation_disagreements'])
    assert.equal(auditRowsHealthy([rows[0],{...rows[1],neural:{[field]:1}}],assets),false);
});
test('a slow complete H200 baseline reaches verified labels, training and bounded screening',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-screening-'));let baselines=0,preparations=0,updates=0,audits=0;
  try{const result=await staged({root,deadline:Date.now()+32*60000,phase:()=>{},execute:async(name,args)=>{
    if(name==='blitz'){
      ++baselines;const out=args[3],assets=JSON.parse(fs.readFileSync(args[1])).assets;
      write(out+'/summary.json',{complete:true,seconds:448});
      for(const a of assets)write(out+'/rows/'+a.id+'.json',{id:a.id,complete:true,neural:{}});
    }else if(name==='blitz-neural-action-prepare'){
      ++preparations;assert.equal(updates,0);
      write(args[1]+'/index.json',{complete:true,reference_confirmed:true,preceding_lod_emitted:true,source_triangles:100,previous_triangles:90});
    }else if(name==='blitz-neural-action-train'){
      ++updates;assert.equal(preparations,4);write(args[1]+'/latest.json',trainedHealth(args));
    }else{
      ++audits;assert.equal(name,'node');assert.equal(args[args.indexOf('--scenario')+1],'screening');
      assert.equal(args[args.indexOf('--action-trials')+1],'8');assert.equal(args[args.indexOf('--neural-confirmation')+1],'gpu');
      assert.ok(!args.includes('--previous'));write(args[1]+'/report.json',{complete:true,gate:{passed:false,reason:'screening_only'},persisted:false});
    }
  }});assert.equal(baselines,1);assert.equal(updates,6);assert.equal(audits,2);
  assert.equal(result.training_steps,6*8192);assert.equal(result.complete,true);
  assert.equal(result.quality_proven,false);assert.equal(result.generalization_gate_passed,false);
  assert.equal(result.stop_reason,'two_screening_checkpoints_complete');
  }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('fresh labels must be confirmed before staged optimizer updates',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-labels-'));let calls=0;
  try{await assert.rejects(generalize({root,deadline:Date.now()+120000,phase:()=>{},diagnostic:true,refresh:true,execute:async(name,args)=>{
    ++calls;assert.equal(name,'blitz-neural-action-prepare');write(args[1]+'/index.json',{complete:true,reference_confirmed:false});
  }}),/Fresh curriculum/);assert.equal(calls,1);}finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('diagnostic training resumes for a second checkpoint and never claims pilot superiority',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-stages-'));let updates=0,audits=0;
  try{const result=await generalize({root,deadline:Date.now()+300000,phase:()=>{},diagnostic:true,execute:async(name,args)=>{
    if(name==='blitz-neural-action-train'){++updates;assert.equal(Number(args[args.indexOf('--steps')+1]),updates<=3?8192:16384);write(args[1]+'/latest.json',trainedHealth(args));}
    else{++audits;assert.equal(name,'node');assert.equal(args[args.indexOf('--neural-confirmation')+1],'gpu');assert.equal(args[args.indexOf('--scenario')+1],'diagnostic');assert.ok(!args.includes('--previous'));write(args[1]+'/report.json',{complete:true,gate:{passed:false,reason:'diagnostic_only'},persisted:false});}
  }});assert.equal(updates,6);assert.equal(audits,2);assert.equal(result.complete,true);assert.equal(result.generalization_gate_passed,false);assert.equal(result.quality_proven,false);
  }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('an incomplete matched pilot prevents a second cloud training stage',async()=>{
  for(const reusePrepared of [false,true]){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-action-'));let updates=0,audits=0,preparations=0;
  try{const result=await generalize({root,deadline:Date.now()+120000,phase:()=>{},reusePrepared,execute:async(name,args)=>{
    if(name==='blitz-neural-action-prepare'){++preparations;write(args[1]+'/index.json',{complete:true,reference_confirmed:true,preceding_lod_emitted:true,source_triangles:100,previous_triangles:90});}
    else if(name==='blitz-neural-action-train'){++updates;write(args[1]+'/latest.json',trainedHealth(args));}
    else{assert.equal(name,'node');++audits;write(args[1]+'/report.json',{complete:false,gate:{passed:false,reason:'incomplete'},persisted:false});}
  }});assert.equal(preparations,reusePrepared?0:6);assert.equal(updates,3);assert.equal(audits,1);assert.equal(JSON.parse(fs.readFileSync(root+'/curriculum/progress.json')).complete,true);assert.equal(result.complete,false);assert.equal(result.generalization_gate_passed,false);assert.equal(result.stop_reason,'incomplete_matched_pilot');
  }finally{fs.rmSync(root,{recursive:true,force:true});}
  }
});
test('matched pilot requires three seeds, all controls, complete assets and measured advantage',()=>{
  const methods=['learned','learned','learned','constant','shuffled','shuffled','shuffled','shortest','current-plane'];
  const runs=methods.map(ranking=>({ranking,code:0,summary:{complete:true,completed:8,expected:8,score:ranking==='learned'?8:5,categories:{a:{count:2,mean_ratio:.8},b:{count:2,mean_ratio:.9}}}}));
  for(const scoreGain of [1,2])assert.equal(pilotDecision(runs,{scoreGain}).passed,true);
  assert.equal(pilotDecision(runs,{scoreGain:4}).passed,false);
  runs[1].health_complete=false;assert.equal(pilotDecision(runs).reason,'incomplete');runs[1].health_complete=true;
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
