import assert from 'node:assert/strict';
import {selectPilot} from '../research/neural/freeze-pilot.mjs';
const assets=['manufactured','organic','rocks','stress'].flatMap(category=>Array.from({length:7},(_,i)=>({id:`${category}-${i}`,category,split:i===6?'held-out':'development',triangles:i===5?150001:100+i,source_group:`${category}-${i===1?0:i}`})));
const rows=selectPilot({assets},{assets});
assert.equal(rows.length,12);assert.equal(new Set(rows.map(a=>a.source_group)).size,12);
for(const category of ['manufactured','organic','rocks','stress'])assert.deepEqual(rows.filter(a=>a.category===category).map(a=>a.triangles),[100,103,104]);
assert.throws(()=>selectPilot({assets:assets.filter(a=>a.category!=='rocks')},{assets}),/three distinct/);
console.log('Frozen pilot selection contracts passed');
const {residentArguments,persistentLearningCycle}=await import('../research/neural/resident-cycle.mjs');
const args=residentArguments('run',{model:'policy',checkpoint:'optimizer',candidateBatch:2,backend:'fused'});
assert.equal(args[args.indexOf('--duration-minutes')+1],'120');assert.equal(args[args.indexOf('--finalize-minutes')+1],'10');
assert.equal(args[args.indexOf('--vertex-storage')+1],'packed');assert.equal(args[args.indexOf('--update-backend')+1],'fused');
assert.throws(()=>residentArguments('run',{model:'m',checkpoint:'o',minutes:119}),/learning window/);
let executed=false;await assert.rejects(()=>persistentLearningCycle({latest:120*60000,execute(){executed=true;}},{},{now:()=>0}),/no longer fit/);assert.equal(executed,false);

assert.equal(args[args.indexOf('--minutes')+1],'130');assert.equal(args[args.indexOf('--training-profile')+1],'coverage');assert.ok(!args.includes('--warmstart'));
const appearance=residentArguments('run',{model:'policy',checkpoint:'optimizer',profile:'attributes'});assert.ok(appearance.includes('--warmstart'));
const {pilotReady}=await import('../research/neural/coverage-validation.mjs');
const pilot={complete:true,conditions:Array.from({length:48},(_,i)=>({complete:true,result:{complete:true,reference_confirmed:true,status:i%4?'complete':'predecessor_unavailable',states:2,category:['manufactured','organic','rocks','stress'][i%4]}}))};
assert.ok(pilotReady(pilot));for(const field of ['complete','reference_confirmed']){const invalid=structuredClone(pilot);invalid.conditions[0].result[field]=false;assert.ok(!pilotReady(invalid));}
for(const status of ['resource_failure','unknown_audit','cancelled','representation_failure']){const invalid=structuredClone(pilot);invalid.conditions[0].result.status=status;assert.ok(!pilotReady(invalid));}
const empty=structuredClone(pilot);for(const row of empty.conditions)if(row.result.category==='stress')row.result.states=0;assert.ok(!pilotReady(empty));
const {pretrainingBudget,pretrainingAuthorization}=await import('../research/neural/action-budget.mjs');
const base=pretrainingAuthorization.baseline_usd;for(const rate of [.6,1.1,2.1]){const quote=pretrainingBudget({billed:base-1,rate,additionalAccrued:base-2.75});assert.ok(quote.maximum_total_usd<=base+8);assert.equal(quote.maximum_rental_usd,3*(rate+.01));}
assert.throws(()=>pretrainingBudget({billed:base+1,rate:2.1,additionalAccrued:base-2.75}),/cap/);assert.throws(()=>pretrainingBudget({billed:base,rate:2.11,additionalAccrued:base-2.75}),/180-minute/);
