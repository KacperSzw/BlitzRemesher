import {test} from 'node:test';
import assert from 'node:assert/strict';
import {widthDecision} from '../research/neural/width-quality.mjs';
const measurement={complete:true,passed:true};
function audit(triangles,seconds=1){return {complete:true,manifest_sha256:'same inputs',settings_sha256:'same gates',rows:['a','b'].map(asset=>({asset,ranking:'learned',status:'complete',seconds,lods:[{triangles:100},{triangles,source:measurement,adjacent:measurement}]}))};}
test('larger policies require matching successful audits for every independent seed',()=>{
  const seeds=[7,19,37],rows=seeds.flatMap(seed=>[{seed,width:64,code:0,audit:audit(80)},{seed,width:128,code:0,audit:audit(64)}]);
  for(const required of [.1,.19])assert.equal(widthDecision(rows,seeds,required,2).selected_width,128);
  assert.equal(widthDecision(rows,seeds,.21,2).selected_width,64);
  for(const change of [a=>a.complete=false,a=>a.settings_sha256='changed',a=>a.rows[0].status='failed',a=>a.rows[1].seconds=9]){
    const bad=structuredClone(rows);change(bad.at(-1).audit);assert.equal(widthDecision(bad,seeds).selected_width,64);
  }
  assert.equal(widthDecision(rows.slice(0,-1),seeds).selected_width,64);
  for(const status of [{code:null,signal:'SIGTERM'},{code:7},{code:undefined},{timed_out:true},{cancelled:true}])for(const index of [0,1]){
    const bad=structuredClone(rows);Object.assign(bad[index],status);const result=widthDecision(bad,seeds);assert.equal(result.selected_width,64);assert.equal(result.passed,false);
  }
});
