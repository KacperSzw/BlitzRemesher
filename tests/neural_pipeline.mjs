import {test} from 'node:test';
import assert from 'node:assert/strict';
import {pipelineValidationBudget,pretrainingAuthorization,actionAccrued} from '../research/neural/action-budget.mjs';
test('validation and forensic rentals share the current pinned grant and reserve',()=>{
  const base=pretrainingAuthorization.baseline_usd;
  for(const minutes of [35,60])for(const rate of [.5,1.1,2.1]){
    const b=pipelineValidationBudget({billed:base+1,additionalAccrued:base-2.75+1,rate,minutes});
    assert.equal(b.cap_usd,base+8);assert.equal(b.reserve_usd,1);assert.equal(b.maximum_rental_usd,minutes/60*(rate+.01));
    assert.ok(b.maximum_total_usd<=b.cap_usd);
  }
  for(const extra of [{billed:base+7},{additionalAccrued:base+7},{minutes:120},{rate:2.5},{billed:0,additionalAccrued:0}])
    assert.throws(()=>pipelineValidationBudget({billed:base,additionalAccrued:base-2.75,rate:1.1,...extra}));
  assert.equal(actionAccrued([{name:'forensic',experiment:'pipeline-validation',started_at:0,terminated_at:3600000,compute_terminated:true,deployment:{gpu_hourly_usd_cap:1.1}}]),1.11);
});
