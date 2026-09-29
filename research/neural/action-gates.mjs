export function endpointDecision(history){
  if(!history.length)return {stop:false,passed:false};
  for(const row of history)if(!row.health?.finite||!row.health.restored||!row.health.optimizer_restored||!row.proof?.complete)
    return {stop:true,passed:false,reason:'health_or_incomplete_audit'};
  if(history.length>=2&&history.slice(-2).every(r=>r.proof.proof_passed))return {stop:true,passed:true,reason:'endpoint_gate_persisted'};
  let best=-Infinity,bestReduction=-Infinity,flat=0;
  for(const {proof} of history){const membership=proof.preferred_membership,reduction=proof.rows[0].reduction;
    if(!Number.isFinite(membership)||!Number.isFinite(reduction))return {stop:true,passed:false,reason:'nonfinite_action_metric'};
    if(membership>best+1e-4||reduction>bestReduction){flat=0;best=Math.max(best,membership);bestReduction=Math.max(bestReduction,reduction);}else ++flat;
  }
  return {stop:flat>=2,passed:false,reason:flat>=2?'two_flat_action_audits':'continue'};
}
