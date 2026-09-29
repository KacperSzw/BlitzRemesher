// Cumulative authorization for the action-learning experiment, in USD.
export function actionAccrued(states,now=Date.now(),hourlyCap=2.51){
  let total=0;const seen=new Set();
  for(const s of states){if(!['action-v2','action-v2-pilot','action-v2-evaluate','action-v2-staged'].includes(s.experiment)||seen.has(s.name))continue;seen.add(s.name);
    const end=s.compute_terminated?s.terminated_at:now;
    if(!s.name||!Number.isFinite(s.started_at)||!Number.isFinite(end)||end<s.started_at)throw new Error('Reconcile incomplete action rental ledger');
    total+=(end-s.started_at)/3600000*hourlyCap;
  }
  return total;
}
// User authorized $8 more on 2026-09-29 after evaluation-01 was collected and
// deleted. Anchor this grant to that conservative ledger, never to the old $10
// ceiling; retries and subsequent rentals consume the same additional grant.
export const continuationAuthorization=Object.freeze({baseline_usd:5.513044952777777,additional_usd:8});
export function continuationBudget(input){
  const {baseline_usd,additional_usd}=continuationAuthorization;
  const result=actionBudget({...input,cap:baseline_usd+additional_usd,priorEstimate:2.75,reserve:1});
  if(result.prior_assumed_usd<baseline_usd-1e-9)throw new Error('Continuation ledger is missing earlier rentals');
  return {...result,authorization:continuationAuthorization,maximum_additional_usd:result.maximum_total_usd-baseline_usd};
}
export function actionBudget({billed,rate,minutes=60,priorEstimate=2.75,additionalAccrued=0,cap=10,reserve=1}){
  for(const value of [billed,rate,minutes,priorEstimate,additionalAccrued,cap,reserve])if(!Number.isFinite(value)||value<0)throw new Error('Invalid experiment budget input');
  if(!rate||minutes<45||minutes>150||reserve<1)throw new Error('Action rental must include bounded setup, proof and collection');
  const prior=Math.max(billed,priorEstimate+additionalAccrued),rental=minutes/60*(rate+.01);
  if(prior+rental+reserve>cap)throw new Error('Cumulative cloud cap would be exceeded');
  return {cap_usd:cap,provider_billed_usd:billed,additional_accrued_upper_usd:additionalAccrued,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,minutes};
}
