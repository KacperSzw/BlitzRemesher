// Exit 2 is resumable elapsed-time work. Fixed resource limits need a changed
// implementation/workload, so do not repeatedly submit the same blocked asset.
export function auditProgress(summary,previous=-1){
  if(summary.blocked_assets?.length){
    const details=summary.blocked_assets.map(asset=>`${asset.id}: ${JSON.stringify(asset.diagnostic??asset.reason)}`).join('; ');
    const error=new Error(`Audit resource limit: ${details}`);error.audit=summary;throw error;
  }
  if(summary.failed_assets?.length){
    const error=new Error(`Audit bake failures: ${JSON.stringify(summary.failed_assets)}`);error.audit=summary;throw error;
  }
  if(summary.complete)return summary.completed;
  if(summary.completed<=previous){const error=new Error('Audit made no progress');error.audit=summary;throw error;}
  return summary.completed;
}
