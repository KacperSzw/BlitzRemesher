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
export function pilotDecision(runs,{scoreGain=1,categories=2}={}){
  if(runs.length!==9||runs.some(r=>r.code!==0||!r.summary?.complete||r.summary.completed!==8||r.summary.expected!==8||!Number.isFinite(r.summary.score)))return {passed:false,reason:'incomplete'};
  const learned=runs.filter(r=>r.ranking==='learned'),controls=runs.filter(r=>['constant','shuffled'].includes(r.ranking));
  if(learned.length!==3||controls.length!==4||runs.filter(r=>r.ranking==='shortest').length!==1||runs.filter(r=>r.ranking==='current-plane').length!==1)return {passed:false,reason:'missing_controls'};
  const strongest=Math.max(...controls.map(r=>r.summary.score));
  const gains=learned.map(r=>r.summary.score-strongest),positive=learned.map(r=>Object.values(r.summary.categories).filter(c=>c.count>0&&c.mean_ratio<1).length);
  return {passed:gains.every(g=>g>=scoreGain)&&positive.every(n=>n>=categories),reason:'measured',strongest_control:strongest,gains,positive_categories:positive};
}
