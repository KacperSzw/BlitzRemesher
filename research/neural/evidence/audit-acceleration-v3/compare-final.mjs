import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
const root='runs/neural/audit-acceleration',runs=[];
const read=p=>JSON.parse(fs.readFileSync(p));
for(const [i,variant] of ['baseline','final','final','baseline'].entries()){
 const binary=variant==='final'?'build/neural/blitz':root+'/blitz-'+variant,out=root+'/final-repeat-'+i+'-'+variant;
 const log=execFileSync(binary,['bench','research/neural/action-diagnostic.json',root+'/profile-config.json',out,'--neural-model','research/neural/evidence/action-curriculum-v2/seed-101-step-8192.blzn','--neural-confirmation','gpu','--action-trials','2','--action-batch','32','--gpu-memory-mib','512','--minutes','2'],{encoding:'utf8'});fs.writeFileSync(out+'.log',log);
 const summary=read(out+'/summary.json'),rows=read('research/neural/action-diagnostic.json').assets.map(a=>read(out+'/rows/'+a.id+'.json'));
 assert.equal(summary.complete,true);
 const outputs=rows.map(r=>({id:r.id,output:r.output_sha256,attributes:r.attributes_sha256,ratio:r.ratio,result:r.result}));
 if(runs.length)assert.deepEqual(outputs,runs[0].outputs);
 runs.push({variant,seconds:summary.seconds,rows:rows.map(r=>({id:r.id,seconds:r.seconds,gpu_audit:r.neural.gpu_audit_seconds})),outputs});
 console.log(variant,summary.seconds);
}
fs.writeFileSync(root+'/final-comparison.json',JSON.stringify({complete:true,identical_results:true,runs},null,2)+'\n');
