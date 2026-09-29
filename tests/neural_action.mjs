import {test} from 'node:test';
import assert from 'node:assert/strict';
import {actionBudget,actionAccrued} from '../research/neural/action-budget.mjs';
import {endpointDecision,pilotDecision} from '../research/neural/action-gates.mjs';
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
