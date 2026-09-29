import fs from 'node:fs/promises';import {createHash} from 'node:crypto';import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');process.chdir(root);const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const report=await read('research/appearance/analysis.json'),replay=await read('research/appearance/replay-manifest.json');
const sha=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const results=[];
for(const r of report.runs.filter(r=>r.id.startsWith('appearance-'))) {
 const assets=[];for(const row of r.rows){let a;const file=`research/appearance/audits/${r.id}/${row.id}.json`;try{a=await read(file);}catch(e){if(e.code!=='ENOENT')throw e;}
  if(a){if(a.run_sha256!==r.metadata.run_sha256||a.chain_bin_sha256!==await sha(`${r.dir}/meshes/${row.id}/chain.bin`)||a.chain_gltf_sha256!==await sha(`${r.dir}/meshes/${row.id}/chain.gltf`)||a.row_sha256!==await sha(`${r.dir}/rows/${row.id}.json`))throw Error('Audit provenance mismatch');}
  assets.push({id:row.id,unchanged:row.fallback,complete:a?.complete??false,passed:a?.passed??false,seed:a?.audit.seed,
   failures:a?.lods.flatMap(l=>['source','adjacent'].filter(k=>!l[k].passed).map(k=>({level:l.level,comparison:k,resource_limited:l[k].resource_limited,error_px:l[k].error_px,nonfinite_error:l[k].nonfinite_error,view:l[k].worst_view,area:l[k].changed_area})))??[]});
 }
 const c=report.comparisons.find(c=>c.candidate===r.id);
 results.push({id:r.id,complete:r.complete,reduction_gate:c?.reduction_gate??false,observed_time_gate:c?Number.isFinite(c.max_time_ratio)&&c.max_time_ratio<=4:false,assets,
  independent_gate:assets.length===8&&assets.every(a=>a.complete&&a.passed),qualified_reduced:assets.filter(a=>!a.unchanged&&a.complete&&a.passed).map(a=>a.id)});
}
const decision={version:1,promoted:false,default:'unchanged',results,paired_timings:'not run: pilot usefulness gate required',validation:'not run: pilot usefulness gate required',held_out:'unused'};
await fs.writeFile('research/appearance/decision.json',JSON.stringify(decision,null,2)+'\n');
let md='# Decision: keep appearance stages experimental\n\n';
md+='The implementation is complete; useful appearance reduction has not been established. Production defaults remain unchanged. All limits and the sampled final-audit policy are preserved.\n\n';
md+='## What the controlled replay established\n\n';
md+=`${replay.observations.filter(x=>x.fail_to_pass).length} fixed-candidate/camera groups change from fail to pass at a higher sample density, and ${replay.observations.filter(x=>x.pass_to_fail).length} change from pass to fail. This demonstrates nonmonotonic sampling, not that a particular density is correct. Stronger 50% triangle targets mostly retain normal-mismatch witnesses. Source screens alone do not explain the stalled chains. [Raw replay](replay-manifest.json).\n\n`;
md+='## Pilot and independent audits\n\n| Stage | Pilot complete | Useful-reduction gate | Observed time ≤4× | Independently qualified reduced chains |\n|---|---|---|---|---:|\n';
for(const r of results)md+=`| ${r.id} | ${r.complete?'Yes':'No'} | ${r.reduction_gate?'Pass':'Not passed'} | ${r.observed_time_gate?'Yes':'No / incomplete'} | ${r.qualified_reduced.length} |\n`;
md+='\nThe position-fitting trial was stopped after six completed assets: its manufactured assets already made the four-category usefulness gate impossible, and the Bell attempt exceeded the bake allowance. Its partial rows and [interruption record](../runs/appearance-position/interruption.json) are retained without an aggregate score.\n';
md+='\nIndividual timing observations cannot establish repeatability. Qualification requires a complete fresh 642+64-camera audit (seed 0xA1172027) of LOD1–7 against source and predecessor. Identity fallbacks pass without providing reduction benefit.\n\n';
for(const r of results)if(r.assets.some(a=>a.complete)) {md+=`### ${r.id}\n\n| Asset | Outcome | Failure witnesses |\n|---|---|---|\n`;for(const a of r.assets){const outcome=a.complete?(a.passed?(a.unchanged?'Pass · unchanged source':'Pass · reduced'):'Fail'):'Incomplete / not audited';const failures=a.failures.map(f=>`LOD${f.level} ${f.comparison}, view ${f.view}: ${f.resource_limited?'resource limit':f.nonfinite_error?'appearance mismatch':f.error_px+' px'}`).join('; ');md+=`| ${a.id} | ${outcome} | ${failures||'—'} |\n`;}}
md+='\nValidation20 and three paired timing repetitions are gated on a useful pilot and are not launched for a failed development gate. Held-out release assets remain unused. No game-ready claim is made for failing or unaudited chains.\n\n';
md+='## Next evidence needed\n\nEvery selected chain in the complete ordering pilot reuses source vertices and adds zero vertex bytes. Ordering and attribute fitting produce byte-identical exported geometry and attributes, so extra fitting work did not reach the delivered chains. The next bounded ablation should examine rebuilt proposal density: the current target clamp divides the available vertex budget by three vertices per triangle, even though actual compact vertex counts decide admission. Compare targets nearer the measured vertex budget while retaining the same cap and independent gates. This is a testable proposal, not evidence that denser candidates will qualify.\n\n[Measurements](REPORT.md) · [Offline board](../../examples/appearance-board/index.html) · [Progress curves](progress.svg) · [Verification](VERIFICATION.md).\n';
await fs.writeFile('research/appearance/DECISION.md',md);console.log(JSON.stringify(results.map(({id,complete,reduction_gate,independent_gate})=>({id,complete,reduction_gate,independent_gate}))));
