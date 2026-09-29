// Run after CTest's IO fixture producer. Never touches a scored run.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..'),dir=resolve(root,'build/chain-search/audit-fixture');
await fs.rm(dir,{recursive:true,force:true});await fs.mkdir(dir+'/rows',{recursive:true});
await fs.cp(resolve(root,'build/release/io-test-output/distinct'),dir+'/meshes/fixture',{recursive:true});
const result=JSON.parse(await fs.readFile(dir+'/meshes/fixture/lods.json','utf8'));
await fs.writeFile(dir+'/metadata.json',JSON.stringify({run_sha256:'fixture',config:{levels:3,profile:'coverage',max_changed_area:.5}}));
await fs.writeFile(dir+'/rows/fixture.json',JSON.stringify({id:'fixture',run_sha256:'fixture',complete:true,result}));
const output=dir+'/audit.json',binary=resolve(root,'build/sanitize/blitz-tail-audit');
const run=()=>execFileSync(binary,[dir,'fixture',output,'0xA1172026','0','1'],{cwd:root,stdio:'pipe'});
run();let audit=JSON.parse(await fs.readFile(output,'utf8'));
assert.equal(audit.complete,true);assert.equal(audit.passed,false);assert.equal(audit.lods.length,2);
assert.equal(audit.lods[0].source.passed,true);assert.equal(audit.lods[1].source.passed,false);
// Simulate an interrupted earlier slot while retaining a resolved later one.
const marker='f'.repeat(64);audit.lods[1].binary_sha256=marker;
for(const reason of ['resource_limited','cancelled']) {
for(const kind of ['source','adjacent'])Object.assign(audit.lods[0][kind],{passed:false,complete:false,resource_limited:false,cancelled:false,[reason]:true,error_px:null,nonfinite_error:true,views:reason==='cancelled'?1:0});
audit.complete=audit.passed=false;await fs.writeFile(output,JSON.stringify(audit));
run();audit=JSON.parse(await fs.readFile(output,'utf8'));
assert.equal(audit.complete,true);assert.equal(audit.passed,false);assert.equal(audit.lods[0].source.passed,true);
assert.equal(audit.lods[1].binary_sha256,marker);assert.equal(audit.stage_seconds.raster,0);
}
const valid=await fs.readFile(output,'utf8');audit.chain_bin_sha256='wrong';await fs.writeFile(output,JSON.stringify(audit));
assert.throws(run,/audit resume provenance mismatch/);
await fs.writeFile(output,valid);
const checks={complete:true,sanitized:true,checks:['cancellation is not a rejection witness','failure witness resolves a comparison','all scheduled levels attempted','resume resolved levels across an interrupted slot','preserve per-level auditor provenance','reject mismatched export hashes']};
await fs.writeFile(resolve(root,process.argv[2]??'research/chain-search/audit-check.json'),JSON.stringify(checks,null,2)+'\n');
console.log(JSON.stringify(checks));
