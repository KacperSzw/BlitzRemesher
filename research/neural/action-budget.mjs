// Cumulative authorization for the action-learning experiment, in USD.
export function actionBudget({billed,rate,minutes=60,priorEstimate=2.75,cap=10,reserve=1}){
  for(const value of [billed,rate,minutes,priorEstimate,cap,reserve])if(!Number.isFinite(value)||value<0)throw new Error('Invalid experiment budget input');
  if(!rate||minutes<45||minutes>150||reserve<1)throw new Error('Action rental must include bounded setup, proof and collection');
  const prior=Math.max(billed,priorEstimate),rental=minutes/60*(rate+.01);
  if(prior+rental+reserve>cap)throw new Error('Cumulative cloud cap would be exceeded');
  return {cap_usd:cap,provider_billed_usd:billed,prior_assumed_usd:prior,reserve_usd:reserve,maximum_rental_usd:rental,maximum_total_usd:prior+rental+reserve,minutes};
}
