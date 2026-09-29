import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
// Parse the real cloud option set, then reject a collection-only fixture before
// any GPU/model work. This catches interface drift without starting training.
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-cli-'));
try{
  const manifest=dir+'/manifest.json';fs.writeFileSync(manifest,JSON.stringify({benchmark_eligible:false,assets:[]}));
  for(const memory of [256,16384])for(const backend of ['cpu','gpu','compare']){
    const result=spawnSync(process.argv[2],['bench',manifest,dir+'/unused.json',dir+'/output','--neural-model',dir+'/unused.blzn','--neural-control','learned','--ranking-seed','7','--action-trials','3','--action-batch','2','--gpu-memory-mib',String(memory),'--neural-confirmation',backend,'--minutes','.1'],{encoding:'utf8',timeout:10000});
    assert.equal(result.status,1,result.stderr);assert.match(result.stderr,/collection-only assets/);
    assert.equal(fs.existsSync(dir+'/output'),false);
  }
  console.log('Cloud benchmark options parsed before the input eligibility gate');
}finally{fs.rmSync(dir,{recursive:true,force:true});}
