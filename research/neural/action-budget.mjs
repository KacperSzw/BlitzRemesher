// Cumulative authorization for the action-learning experiment, in USD.
export function actionAccrued(states,now=Date.now(),hourlyCap=2.51){
  let total=0;const seen=new Set();
  for(const s of states){if(!['action-v2','action-v2-pilot','action-v2-evaluate','action-v2-staged','gpu-refactor','hardware-validation','pipeline-validation'].includes(s.experiment)||seen.has(s.name))continue;seen.add(s.name);
    const end=s.compute_terminated?s.terminated_at:now;
    if(!s.name||!Number.isFinite(s.started_at)||!Number.isFinite(end)||end<s.started_at)throw new Error('Reconcile incomplete action rental ledger');
    const rate=s.deployment?.gpu_hourly_usd_cap===undefined?hourlyCap:s.deployment.gpu_hourly_usd_cap+.01;
    if(!Number.isFinite(rate)||rate<=0)throw new Error('Reconcile invalid action rental rate');
    total+=(end-s.started_at)/3600000*rate;
  }
  return total;
}
// User authorized $8 more on 2026-09-29 after evaluation-01 was collected and
// deleted. Anchor this grant to that conservative ledger, never to the old $10
// ceiling; retries and subsequent rentals consume the same additional grant.
export const continuationAuthorization=Object.freeze({baseline_usd:5.513044952777777,additional_usd:8});
// A separate, explicit $8 grant after all previous rentals were reconciled.
export const refactorAuthorization=Object.freeze({baseline_usd:11.25963675,additional_usd:8});
// Latest user grant replaces the previous remaining allowance. This reconciled
// baseline is pinned once; every validation attempt and retry spends this grant.
export const pretrainingAuthorization=Object.freeze({baseline_usd:17.292720997222222,additional_usd:8});
export function pipelineValidationBudget({billed,rate,additionalAccrued,minutes=60}){
  for(const v of [billed,rate,additionalAccrued,minutes])if(!Number.isFinite(v)||v<0)throw new Error('Invalid pipeline validation budget');
  if(!rate||rate>2.10||![35,60].includes(minutes))throw new Error('Pipeline validation requires a bounded 35 or 60 minute profile');
  const prior=Math.max(billed,2.75+additionalAccrued),rental=minutes/60*(rate+.01),reserve=1,cap=pretrainingAuthorization.baseline_usd+pretrainingAuthorization.additional_usd;
  if(prior<pretrainingAuthorization.baseline_usd-1e-9||prior+rental+reserve>cap)throw new Error('Existing pretraining grant would be exceeded');
  return {authorization:pretrainingAuthorization,cap_usd:cap,provider_billed_usd:billed,additional_accrued_upper_usd:additionalAccrued,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,minutes};
}
export function pretrainingBudget({billed,rate,additionalAccrued,minutes=180}){
  for(const v of [billed,rate,additionalAccrued,minutes])if(!Number.isFinite(v)||v<0)throw new Error('Invalid pretraining budget');
  if(!rate||rate>2.10||minutes!==180)throw new Error('Pretraining requires the bounded 180-minute profile');
  const prior=Math.max(billed,2.75+additionalAccrued),rental=minutes/60*(rate+.01),reserve=1,cap=pretrainingAuthorization.baseline_usd+pretrainingAuthorization.additional_usd;
  if(prior<pretrainingAuthorization.baseline_usd-1e-9||prior+rental+reserve>cap)throw new Error('Pretraining additional $8 cap would be exceeded');
  return {authorization:pretrainingAuthorization,cap_usd:cap,provider_billed_usd:billed,additional_accrued_upper_usd:additionalAccrued,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,minutes};
}
export function hardwareValidationBudget({billed,rate,additionalAccrued,minutes=35}){
  for(const v of [billed,rate,additionalAccrued,minutes])if(!Number.isFinite(v)||v<0)throw new Error('Invalid hardware validation budget');
  if(!rate||rate>2.5||minutes!==35)throw new Error('Hardware validation requires the bounded 35-minute profile');
  const prior=Math.max(billed,2.75+additionalAccrued),rental=minutes/60*(rate+.01),reserve=1,cap=refactorAuthorization.baseline_usd+refactorAuthorization.additional_usd;
  if(prior<refactorAuthorization.baseline_usd-1e-9||prior+rental+reserve>cap)throw new Error('Existing additional cloud cap would be exceeded');
  return {authorization:refactorAuthorization,cap_usd:cap,provider_billed_usd:billed,additional_accrued_upper_usd:additionalAccrued,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,minutes};
}
export function refactorBudget({billed,rate,additionalAccrued,minutes=160}){
  const {baseline_usd,additional_usd}=refactorAuthorization;
  for(const v of [billed,rate,additionalAccrued,minutes])if(!Number.isFinite(v)||v<0)throw new Error('Invalid refactor budget input');
  if(!rate||rate>2.50||minutes!==160)throw new Error('Refactor requires the bounded 160-minute full-cycle profile');
  const prior=Math.max(billed,2.75+additionalAccrued),rental=minutes/60*(rate+.01),reserve=1;
  if(prior<baseline_usd-1e-9)throw new Error('Refactor ledger is missing earlier rentals');
  if(prior+rental+reserve>baseline_usd+additional_usd)throw new Error('Additional cloud cap would be exceeded');
  return {authorization:refactorAuthorization,cap_usd:baseline_usd+additional_usd,provider_billed_usd:billed,additional_accrued_upper_usd:additionalAccrued,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,maximum_additional_usd:prior+rental+reserve-baseline_usd,minutes};
}
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
