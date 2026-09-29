// Cumulative authorization for the action-learning experiment, in USD.
export function actionAccrued(states,now=Date.now(),hourlyCap=2.51){
  let total=0;const seen=new Set();
  for(const s of states){if(!['action-v2','action-v2-pilot','action-v2-evaluate'].includes(s.experiment)||seen.has(s.name))continue;seen.add(s.name);
    const end=s.compute_terminated?s.terminated_at:now;
    if(!s.name||!Number.isFinite(s.started_at)||!Number.isFinite(end)||end<s.started_at)throw new Error('Reconcile incomplete action rental ledger');
    total+=(end-s.started_at)/3600000*hourlyCap;
  }
  return total;
}
export function actionBudget({billed,rate,minutes=60,priorEstimate=2.75,additionalAccrued=0,cap=10,reserve=1}){
  for(const value of [billed,rate,minutes,priorEstimate,additionalAccrued,cap,reserve])if(!Number.isFinite(value)||value<0)throw new Error('Invalid experiment budget input');
  if(!rate||minutes<45||minutes>150||reserve<1)throw new Error('Action rental must include bounded setup, proof and collection');
  const prior=Math.max(billed,priorEstimate+additionalAccrued),rental=minutes/60*(rate+.01);
  if(prior+rental+reserve>cap)throw new Error('Cumulative cloud cap would be exceeded');
  return {cap_usd:cap,provider_billed_usd:billed,additional_accrued_upper_usd:additionalAccrued,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,minutes};
}
