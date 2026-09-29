import {test} from 'node:test';
import assert from 'node:assert/strict';
import {auditProgress} from '../research/neural/audit.mjs';
test('resource limits identify the asset immediately and retain the unscored summary',()=>{
  for(const [id,kind] of [['tree','sample_count'],['rock','workspace_memory']]){
    const summary={complete:false,completed:3,expected:8,score:null,blocked_assets:[{id,diagnostic:{kind,requested:100,limit:64}}]};
    assert.throws(()=>auditProgress(summary),error=>error.audit===summary&&error.message.includes(id)&&error.message.includes(kind));
    assert.equal(summary.score,null);
  }
});
test('deadline progress resumes; stagnation stops; a complete audit finishes',()=>{
  assert.equal(auditProgress({complete:false,completed:2},1),2);
  assert.equal(auditProgress({complete:false,completed:0}),0);
  assert.throws(()=>auditProgress({complete:false,completed:2},2),/no progress/);
  assert.equal(auditProgress({complete:true,completed:8,score:0},8),8);
});
test('a processed batch containing bake exceptions cannot pass readiness',()=>{
  const summary={complete:true,completed:8,score:0,failed_assets:[{id:'mesh',failure:'invalid output'}]};
  assert.throws(()=>auditProgress(summary),error=>error.audit===summary&&error.message.includes('invalid output'));
});
