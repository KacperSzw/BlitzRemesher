import {test} from 'node:test';
import assert from 'node:assert/strict';
import {actionBudget} from '../research/neural/action-budget.mjs';
import {endpointDecision} from '../research/neural/action-gates.mjs';
test('cumulative rental cap includes previous spend, storage allowance and reserve',()=>{
  for(const rate of [1.8,2.1,2.5])for(const billed of [1.9,2.75,4.2]){
    const b=actionBudget({rate,billed,minutes:60});assert.ok(b.maximum_total_usd<=10);assert.ok(b.prior_assumed_usd>=billed);assert.ok(b.reserve_usd>=1);
    assert.equal(b.maximum_rental_usd,rate+.01);
  }
  for(const invalid of [{billed:9,rate:2},{billed:2,rate:5,minutes:150},{billed:NaN,rate:2},{billed:2,rate:2,minutes:Infinity},{billed:2,rate:2,reserve:0}])assert.throws(()=>actionBudget(invalid));
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
