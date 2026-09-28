import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile, readdir, mkdir, writeFile} from 'node:fs/promises';
import {basename, join} from 'node:path';

const args=process.argv.slice(2);
if(![4,6,7,8,9].includes(args.length))throw Error('Usage: node make-report.mjs UNCAPPED_RUN CAP_RUN ADAPTIVE_RUN OUTPUT_DIR [VALIDATION_UNCAPPED_RUN VALIDATION_CAP_RUN [VALIDATION_ADAPTIVE_RUN] [FALLBACK_CONTROL_VALIDATION_RUN FALLBACK_VALIDATION_RUN]]');
const read=async path=>JSON.parse(await readFile(path,'utf8'));
const pct=value=>value===null?'—':(100*value).toFixed(2)+'%';
const fixed=value=>value===null?'—':value.toFixed(2);
const precise=value=>value===null?'—':value.toFixed(4);
const location=value=>value?pct(value.value)+' (LOD '+value.level+', view '+value.view+')':'—';
const field=(row,key)=>row?.[key]??null;
const sha256=async path=>createHash('sha256').update(await readFile(path)).digest('hex');
function balanced(rows,value) {
  const categories=Map.groupBy(rows,row=>row.category);
  return [...categories.values()].reduce((sum,group)=>sum+group.reduce((n,row)=>n+value(row),0)/group.length,0)/categories.size;
}

function worst(lods,kind) {
  let best=null;
  for(let level=1;level<lods.length;level++) {
    const m=lods[level][kind];
    if(!best || m.changed_area>best.value)best={value:m.changed_area,level,view:m.changed_area_worst_view};
  }
  return best;
}

function inspectRow(row,limit) {
  const failed=row.failed===true || row.complete!==true || !row.result?.lods?.length;
  const lods=failed?[]:row.result.lods;
  const violations=[];
  for(let level=1;level<lods.length;level++)for(const kind of ['source','adjacent']) {
    const m=lods[level][kind],pixels=lods[level][kind==='source'?'source_limit':'transition_limit'];
    if(!m.complete || !m.passed || m.error_px>pixels+1e-12 || m.changed_area>limit+1e-12)
      violations.push({level,kind,error_px:m.error_px,pixel_limit:pixels,changed_area:m.changed_area,area_limit:limit,
        worst_view:m.worst_view,changed_area_worst_view:m.changed_area_worst_view,complete:m.complete,passed:m.passed});
  }
  const source=lods[0]?.triangles??null,final=lods.at(-1)?.triangles??null;
  return {id:row.id,category:row.category,complete:row.complete===true,failed:row.failed===true,
    failure:field(row,'failure'),fallback:row.fallback===true,unreduced:source!==null && source===final,
    output_sha256:field(row,'output_sha256'),attributes_sha256:field(row,'attributes_sha256'),
    canonical_attributes_sha256:field(row,'canonical_attributes_sha256'),
    source_triangles:source,final_triangles:final,final_ratio:source?final/source:null,
    chain_ratio:field(row,'ratio'),source_area:worst(lods,'source'),adjacent_area:worst(lods,'adjacent'),
    source_limits:lods.slice(1).map(lod=>lod.source_limit),
    adjacent_limits:lods.slice(1).map(lod=>lod.transition_limit),
    final_source_area:lods.at(-1)?.source?.changed_area??null,
    resident_bytes:row.result?.storage?.total_bytes??null,
    generation_seconds:field(row,'generation_seconds'),peak_rss_kib:field(row,'peak_rss_kib'),
    area_only_audit_rejections:{source:row.result?.rejections?.source_audit?.area_only_count??0,
      adjacent:row.result?.rejections?.adjacent_audit?.area_only_count??0},
    audit_rejections:{source:row.result?.rejections?.source_audit?.count??0,
      adjacent:row.result?.rejections?.adjacent_audit?.count??0},
    search_rejections:{source:row.result?.rejections?.source_search?.count??0,
      adjacent:row.result?.rejections?.adjacent_search?.count??0},
    candidate_evaluations:row.result?.candidate_evaluations??null,
    topology_fallback_proposals:row.result?.proposal_diagnostics?.topology_fallback_proposals??0,
    violations};
}

async function openRun(path) {
  const meta=await read(join(path,'metadata.json')),summary=await read(join(path,'summary.json'));
  assert.equal(summary.run_sha256,meta.run_sha256,'summary/run hash mismatch: '+path);
  const rowNames=(await readdir(join(path,'rows'))).filter(name=>name.endsWith('.json')).sort();
  const rows=[];
  for(const name of rowNames) {
    const row=await read(join(path,'rows',name));
    assert.equal(row.run_sha256,meta.run_sha256,'row/run hash mismatch: '+name);
    rows.push(inspectRow(row,meta.config.max_changed_area));
  }
  if(summary.complete) {
    assert.equal(rows.length,summary.expected,'complete batch row count: '+path);
    assert.ok(rows.every(row=>row.complete),'complete batch has incomplete row: '+path);
  }
  const failures=rows.filter(row=>row.failed);
  const violations=rows.flatMap(row=>row.violations.map(v=>({id:row.id,...v})));
  const data={name:basename(path),path,run_sha256:meta.run_sha256,binary_sha256:meta.binary_sha256,
    manifest_sha256:meta.manifest_sha256,protocol_sha256:meta.protocol_sha256,
    camera_sha256:meta.camera_sha256,
    config_sha256:meta.config_sha256,build_stamp:meta.build??null,split:meta.split,
    config:meta.config,summary,rows,failures:failures.map(row=>row.id),violations,
    all_delivered_audits_pass:summary.complete && failures.length===0 && violations.length===0,
    category_balanced_final_ratio:summary.complete?balanced(rows,row=>row.final_ratio??1):null};
  if(summary.complete) {
    const score=100*(1-balanced(rows,row=>row.chain_ratio));
    assert.ok(Math.abs(score-summary.score)<1e-8,'SCORE does not match row ratios: '+path);
  }
  return data;
}

function comparable(a,b,validation=false) {
  if(!validation)assert.equal(a.binary_sha256,b.binary_sha256,'comparison requires identical executable');
  assert.equal(a.protocol_sha256,b.protocol_sha256,'comparison requires identical protocol');
  if(!validation)assert.equal(a.manifest_sha256,b.manifest_sha256,'comparison requires identical pilot manifest');
  const config=run=>{const copy=structuredClone(run.config);delete copy.max_changed_area;copy.research.adaptive_targets=false;return copy;};
  assert.deepEqual(config(a),config(b),'comparison changes an uncontrolled setting');
}

const [uncapped,cap,adaptive]=await Promise.all(args.slice(0,3).map(openRun));
comparable(uncapped,cap);comparable(cap,adaptive);
assert.equal(uncapped.config.max_changed_area,1);
assert.equal(cap.config.max_changed_area,0.5);
assert.equal(adaptive.config.max_changed_area,0.5);
assert.equal(uncapped.config.research.adaptive_targets,false);
assert.equal(cap.config.research.adaptive_targets,false);
assert.equal(adaptive.config.research.adaptive_targets,true);
for(const run of [uncapped,cap,adaptive]) {
  assert.equal(run.split,'development','pilot uses the wrong split');
  assert.equal(run.summary.expected,8,'pilot must contain eight frozen assets');
}
const sameAssets=run=>run.rows.map(row=>row.id).join('\n');
assert.equal(sameAssets(uncapped),sameAssets(cap),'pilot asset sets differ');
assert.equal(sameAssets(cap),sameAssets(adaptive),'pilot asset sets differ');

async function oldReplay() {
  const path='research/runs/area-v3-old-uncapped';
  let meta,summary;
  try {[meta,summary]=await Promise.all([read(join(path,'metadata.json')),read(join(path,'summary.json'))]);}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  const fields=['output_sha256','attributes_sha256','canonical_attributes_sha256'];
  const mismatches=[];
  for(const asset of uncapped.rows) {
    const [oldRow,newRow]=await Promise.all([read(join(path,'rows',asset.id+'.json')),read(join(uncapped.path,'rows',asset.id+'.json'))]);
    for(const key of fields)if(oldRow[key]!==newRow[key])mismatches.push({id:asset.id,field:key,old:oldRow[key],new:newRow[key]});
  }
  return {run:path,run_sha256:meta.run_sha256,binary_sha256:meta.binary_sha256,
    complete:summary.complete,assets:uncapped.rows.length,fields,mismatches,identical:summary.complete && mismatches.length===0};
}
const compatibility=await oldReplay();

async function finalTreeReplay() {
  const path='research/runs/area-v3-final-tree-replay',id='ph_dead_quiver_trunk';
  let meta,row;
  try {[meta,row]=await Promise.all([read(join(path,'metadata.json')),read(join(path,'rows',id+'.json'))]);}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  const frozen=await read(join(cap.path,'rows',id+'.json'));
  const fields=['output_sha256','attributes_sha256','canonical_attributes_sha256'];
  const mismatches=fields.filter(key=>row[key]!==frozen[key]);
  return {run:path,run_sha256:meta.run_sha256,binary_sha256:meta.binary_sha256,
    max_changed_area:row.result?.max_changed_area??null,fields,mismatches,
    identical:row.complete && !row.failed && mismatches.length===0};
}
const finalReplay=await finalTreeReplay();

async function denseTail(path) {
  let data;
  try {data=await read(path);}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  assert.equal(data.run_sha256,cap.run_sha256,'dense tail belongs to a different capped run');
  const failures=[];
  for(const lod of data.lods)for(const kind of ['source','adjacent'])if(!lod[kind].passed)
    failures.push({level:lod.level,kind,changed_area:lod[kind].changed_area,area_limit:lod.max_changed_area,
      error_px:lod[kind].error_px,pixel_limit:lod[kind==='source'?'source_limit':'transition_limit'],
      changed_area_worst_view:lod[kind].changed_area_worst_view,views_evaluated:lod[kind].views});
  return {run_sha256:data.run_sha256,binary_sha256:data.binary_sha256,audit:data.audit,passed:data.passed,
    seconds:data.seconds,failures,lods:data.lods};
}
const [dense,denseDefault]=await Promise.all([
  denseTail('research/area-v3/dense-tree-cap-0.5.json'),
  denseTail('research/area-v3/dense-tree-default-seed-cap-0.5.json')]);
if(dense && denseDefault) {
  assert.equal(dense.binary_sha256,denseDefault.binary_sha256,'dense checks use different executables');
  for(const key of ['orthographic','perspective','supersample','max_supersample','max_changed_area'])
    assert.equal(dense.audit[key],denseDefault.audit[key],'dense checks differ beyond seed');
  assert.notEqual(dense.audit.seed,denseDefault.audit.seed,'dense rotations are identical');
}

async function exploratoryCaps() {
  let lower,lowest,audit;
  try {[lower,lowest,audit]=await Promise.all([
    read('research/runs/area-v3-dev-tree-cap-0.45/rows/ph_dead_quiver_trunk.json'),
    read('research/runs/area-v3-dev-tree-cap-0.4/rows/ph_dead_quiver_trunk.json'),
    read('research/area-v3/dense-tree-dev-cap-0.4-rotated.json')]);}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  const frozen=await read(join(cap.path,'rows/ph_dead_quiver_trunk.json'));
  return {scope:'One development tree only; exploratory, unscored, no preset selection',
    cap_045:{same_output_as_frozen_05:lower.output_sha256===frozen.output_sha256,
      final_triangles:lower.result.lods.at(-1).triangles,configured_source_area:lower.result.lods.at(-1).source.changed_area},
    cap_04:{same_output_as_frozen_05:lowest.output_sha256===frozen.output_sha256,
      final_triangles:lowest.result.lods.at(-1).triangles,configured_source_area:lowest.result.lods.at(-1).source.changed_area,
      rotated_dense_passed:audit.passed,rotated_lod7_source_area:audit.lods.at(-1).source.changed_area,
      rotated_lod7_area_limit:audit.lods.at(-1).max_changed_area,
      rotated_lod7_worst_view:audit.lods.at(-1).source.changed_area_worst_view}};
}
const exploratory=await exploratoryCaps();

let validation=null;
if(args.length>=6) {
  const coreValidationCount=args.length===7||args.length===9?3:2;
  const [validationUncapped,validationCap,validationAdaptive]=await Promise.all(args.slice(4,4+coreValidationCount).map(openRun));
  comparable(uncapped,validationUncapped,true);comparable(cap,validationCap,true);
  comparable(validationUncapped,validationCap);
  assert.equal(sameAssets(validationUncapped),sameAssets(validationCap),'validation asset sets differ');
  for(const run of [validationUncapped,validationCap,...(validationAdaptive?[validationAdaptive]:[])]) {
    assert.equal(run.split,'validation','validation uses the wrong split');
    assert.equal(run.summary.expected,20,'validation must contain 20 frozen assets');
  }
  if(validationAdaptive) {
    comparable(adaptive,validationAdaptive,true);comparable(validationCap,validationAdaptive);
    assert.equal(sameAssets(validationCap),sameAssets(validationAdaptive),'validation asset sets differ');
  }
  validation={uncapped:validationUncapped,cap:validationCap,adaptive:validationAdaptive??null};
}
async function validationComparison() {
  if(!validation)return null;
  const rows=validation.cap.rows.map(capped=>{
    const uncapped=validation.uncapped.rows.find(row=>row.id===capped.id);
    return {id:capped.id,category:capped.category,uncapped_final_triangles:uncapped.final_triangles,
      capped_final_triangles:capped.final_triangles,uncapped_final_source_area:uncapped.final_source_area,
      capped_final_source_area:capped.final_source_area,
      identical_output_and_attributes:uncapped.output_sha256===capped.output_sha256 &&
        uncapped.attributes_sha256===capped.attributes_sha256,
      identical_canonical_attributes:uncapped.canonical_attributes_sha256===capped.canonical_attributes_sha256};
  });
  const breaches=[];
  for(const row of validation.uncapped.rows) {
    const raw=await read(join(validation.uncapped.path,'rows',row.id+'.json'));
    for(let level=1;level<raw.result.lods.length;level++)for(const kind of ['source','adjacent'])
      if(raw.result.lods[level][kind].changed_area>0.5)
        breaches.push({id:row.id,level,kind,changed_area:raw.result.lods[level][kind].changed_area});
  }
  return {rows,changed_rows:rows.filter(row=>!row.identical_output_and_attributes),
    identical_output_and_attributes:rows.filter(row=>row.identical_output_and_attributes).length,
    identical_canonical_attributes:rows.filter(row=>row.identical_canonical_attributes).length,
    no_final_source_area_worsened:rows.every(row=>row.capped_final_source_area<=row.uncapped_final_source_area+1e-12),
    uncapped_area_breaches:breaches,
    capped_area_only_audit_rejections:{source:validation.cap.rows.reduce((n,row)=>n+row.area_only_audit_rejections.source,0),
      adjacent:validation.cap.rows.reduce((n,row)=>n+row.area_only_audit_rejections.adjacent,0)},
    capped_worst_adjacent_area:Math.max(0,...validation.cap.rows.map(row=>row.adjacent_area?.value??0))};
}
const validationChange=await validationComparison();

async function postHocTopology() {
  let run;
  try {run=await openRun('research/runs/area-v3-cap-0.5-topology-relaxed');}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  assert.equal(run.binary_sha256,cap.binary_sha256,'post-hoc D must use the frozen pilot binary');
  assert.equal(run.protocol_sha256,cap.protocol_sha256,'post-hoc D protocol mismatch');
  assert.equal(run.manifest_sha256,cap.manifest_sha256,'post-hoc D manifest mismatch');
  assert.equal(run.split,'development');assert.equal(run.summary.expected,8);
  assert.equal(sameAssets(run),sameAssets(cap),'post-hoc D asset sets differ');
  const normalized=structuredClone(run.config);normalized.objective='quadric';
  assert.deepEqual(normalized,cap.config,'post-hoc D changes more than objective');
  assert.equal(run.config.objective,'topology_relaxed');
  return {run,score_gain_vs_cap:run.summary.score===null?null:run.summary.score-cap.summary.score,
    per_asset:run.rows.map(row=>({id:row.id,cap:cap.rows.find(other=>other.id===row.id),topology_relaxed:row}))};
}
const postHoc=await postHocTopology();
async function readPostHocDense(path) {
  let data;
  try {data=await read(path);}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  assert.ok(postHoc,'topology dense audit has no completed D run');
  assert.equal(data.run_sha256,postHoc.run.run_sha256,'topology dense audit belongs to a different D run');
  return {run_sha256:data.run_sha256,auditor_binary_sha256:data.binary_sha256,audit:data.audit,
    passed:data.passed,lods:data.lods,seconds:data.seconds};
}
const [hatchDense,bellDense,rockDense,treeDense]=await Promise.all([
  readPostHocDense('research/area-v3/dense-hatch-topology-relaxed.json'),
  readPostHocDense('research/area-v3/dense-bell-topology-relaxed.json'),
  readPostHocDense('research/area-v3/dense-rock-topology-relaxed.json'),
  readPostHocDense('research/area-v3/dense-tree-topology-relaxed.json')]);
const dDense={hatch:hatchDense,bell:bellDense,rock:rockDense,tree:treeDense};
for(const item of [bellDense,rockDense,treeDense].filter(Boolean))if(hatchDense) {
  assert.equal(item.auditor_binary_sha256,hatchDense.auditor_binary_sha256,'D dense auditor mismatch');
  assert.deepEqual(item.audit,hatchDense.audit,'D dense contract mismatch');
}

function comparableTopologyFallback(control,experimental,{sameBinary=true,sameManifest=true}={}) {
  if(sameBinary)assert.equal(control.binary_sha256,experimental.binary_sha256,'topology fallback comparison requires identical executable');
  if(sameManifest)assert.equal(control.manifest_sha256,experimental.manifest_sha256,'topology fallback comparison requires identical manifest');
  assert.equal(control.protocol_sha256,experimental.protocol_sha256,'topology fallback protocol mismatch');
  assert.equal(control.camera_sha256,experimental.camera_sha256,'topology fallback camera mismatch');
  assert.equal(control.config.research.topology_fallback??false,false,'control enables topology fallback');
  assert.equal(experimental.config.research.topology_fallback,true,'experimental run disables topology fallback');
  const normalized=run=>{const config=structuredClone(run.config);config.research.topology_fallback=false;return config;};
  assert.deepEqual(normalized(control),normalized(experimental),'topology fallback comparison changes another setting');
  assert.equal(sameAssets(control),sameAssets(experimental),'topology fallback asset sets differ');
}

async function postHocFallback() {
  const path='research/runs/area-v3-cap-0.5-fallback-v2-e';
  const [run,control,historical]=await Promise.all([
    openRun(path),
    openRun('research/runs/area-v3-cap-0.5-fallback-v2-b'),
    openRun('research/runs/area-v3-cap-0.5-topology-fallback')]);
  comparableTopologyFallback(cap,run,{sameBinary:false});
  assert.notEqual(run.binary_sha256,cap.binary_sha256,'E pilot must identify its different executable');
  assert.equal(run.split,'development');assert.equal(run.summary.expected,8);
  assert.ok(run.summary.complete,'post-hoc E pilot is incomplete');
  comparableTopologyFallback(control,run);
  assert.equal(control.split,'development');assert.equal(control.summary.expected,8);
  assert.ok(control.summary.complete,'same-binary E pilot control is incomplete');
  assert.equal(historical.split,'development');assert.equal(historical.summary.expected,8);
  assert.ok(historical.summary.complete,'historical E pilot is incomplete');
  assert.equal(sameAssets(historical),sameAssets(run),'historical E pilot asset set differs');
  const rows=run.rows.map(experimental=>{
    const baseline=control.rows.find(row=>row.id===experimental.id);
    return {id:experimental.id,category:experimental.category,control:baseline,experimental,
      identical_output_and_attributes:baseline.output_sha256===experimental.output_sha256 &&
        baseline.attributes_sha256===experimental.attributes_sha256,
      identical_canonical_attributes:baseline.canonical_attributes_sha256===experimental.canonical_attributes_sha256};
  });
  const replay=control.rows.map(row=>{
    const frozen=cap.rows.find(other=>other.id===row.id);
    return {id:row.id,identical_output_and_attributes:row.output_sha256===frozen.output_sha256 &&
      row.attributes_sha256===frozen.attributes_sha256};
  });
  const historicalReplay=run.rows.map(row=>{
    const prior=historical.rows.find(other=>other.id===row.id);
    return {id:row.id,identical_output_and_attributes:row.output_sha256===prior.output_sha256 &&
      row.attributes_sha256===prior.attributes_sha256,
      identical_canonical_attributes:row.canonical_attributes_sha256===prior.canonical_attributes_sha256};
  });
  assert.ok(replay.every(row=>row.identical_output_and_attributes),'corrected v2 B control changed frozen B outputs');
  assert.ok(historicalReplay.every(row=>row.identical_output_and_attributes && row.identical_canonical_attributes),
    'corrected v2 E changed historical E outputs');
  const historicalDenseExports={};
  for(const id of ['si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7',
    'si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652']) {
    historicalDenseExports[id]={};
    for(const file of ['chain.bin','chain.gltf']) {
      const before=await sha256(join(historical.path,'meshes',id,file));
      const corrected=await sha256(join(run.path,'meshes',id,file));
      assert.equal(corrected,before,'corrected E dense-target export differs from historical E');
      historicalDenseExports[id][file]=corrected;
    }
  }
  const treeId='ph_dead_quiver_trunk';
  const treeExport={};
  for(const file of ['chain.bin','chain.gltf']) {
    treeExport[file]={control_sha256:await sha256(join(cap.path,'meshes',treeId,file)),
      experimental_sha256:await sha256(join(run.path,'meshes',treeId,file))};
  }
  const treeByteIdentical=Object.values(treeExport).every(item=>item.control_sha256===item.experimental_sha256);
  assert.ok(treeByteIdentical,'E tree export changed; inherited B dense result is invalid');
  return {run,control,strict_same_binary_comparison:true,
    score_gain_vs_control:run.summary.score-control.summary.score,
    different_binary_from_frozen_cap:run.binary_sha256!==cap.binary_sha256,
    frozen_control_replay_identical:replay.filter(row=>row.identical_output_and_attributes).length,
    frozen_control_replay_rows:replay,
    historical_e:{run:'research/runs/area-v3-cap-0.5-topology-fallback',run_sha256:historical.run_sha256,
      binary_sha256:historical.binary_sha256,score:historical.summary.score,
      identical_output_and_attributes:historicalReplay.filter(row=>row.identical_output_and_attributes).length,
      identical_canonical_attributes:historicalReplay.filter(row=>row.identical_canonical_attributes).length,
      replay_rows:historicalReplay,dense_target_exports:historicalDenseExports},
    rows,changed_rows:rows.filter(row=>!row.identical_output_and_attributes),
    identical_output_and_attributes:rows.filter(row=>row.identical_output_and_attributes).length,
    identical_canonical_attributes:rows.filter(row=>row.identical_canonical_attributes).length,
    topology_fallback_proposals:run.rows.reduce((n,row)=>n+row.topology_fallback_proposals,0),
    candidate_evaluations:{control:control.rows.reduce((n,row)=>n+(row.candidate_evaluations??0),0),
      experimental:run.rows.reduce((n,row)=>n+(row.candidate_evaluations??0),0)},
    tree_export:treeExport,tree_byte_identical:treeByteIdentical};
}
const postHocFallbackPilot=await postHocFallback();

async function readFallbackDense(path) {
  let data;
  try {data=await read(path);}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  assert.ok(postHocFallbackPilot,'E dense audit has no completed pilot');
  assert.equal(data.run_sha256,postHocFallbackPilot.run.run_sha256,'E dense audit belongs to another run');
  if(dense)assert.deepEqual(data.audit,dense.audit,'E/B dense contracts differ');
  return {run_sha256:data.run_sha256,auditor_binary_sha256:data.binary_sha256,audit:data.audit,
    passed:data.passed,lods:data.lods,seconds:data.seconds};
}
const [fallbackHatchDense,fallbackBellDense,fallbackTreeDense]=await Promise.all([
  readFallbackDense('research/area-v3/dense-hatch-topology-fallback-v2.json'),
  readFallbackDense('research/area-v3/dense-bell-topology-fallback-v2.json'),
  readFallbackDense('research/area-v3/dense-tree-topology-fallback-v2.json')]);
for(const item of [fallbackBellDense,fallbackTreeDense].filter(Boolean))if(fallbackHatchDense)
  assert.equal(fallbackHatchDense.auditor_binary_sha256,item.auditor_binary_sha256,'E dense auditor mismatch');

let postHocFallbackValidation=null;
if(args.length>=8) {
  assert.ok(postHocFallbackPilot,'E validation requires its archived pilot');
  const [control,experimental]=await Promise.all(args.slice(-2).map(openRun));
  comparableTopologyFallback(control,experimental);
  assert.equal(control.split,'validation');assert.equal(experimental.split,'validation');
  assert.equal(control.summary.expected,20);assert.equal(experimental.summary.expected,20);
  assert.ok(control.summary.complete && experimental.summary.complete,'E validation batches must be complete');
  assert.equal(control.binary_sha256,postHocFallbackPilot.run.binary_sha256,'E validation does not use corrected v2 pilot executable');
  assert.deepEqual(experimental.config,postHocFallbackPilot.run.config,'E validation config differs from E pilot');
  assert.equal(control.manifest_sha256,validation.cap.manifest_sha256,'E validation differs from frozen validation manifest');
  assert.equal(control.protocol_sha256,validation.cap.protocol_sha256,'E validation differs from frozen protocol');
  assert.equal(sameAssets(control),sameAssets(validation.cap),'E validation asset set differs from prior validation');
  const rows=experimental.rows.map(row=>{
    const baseline=control.rows.find(other=>other.id===row.id);
    return {id:row.id,category:row.category,control:baseline,experimental:row,
      identical_output_and_attributes:baseline.output_sha256===row.output_sha256 &&
        baseline.attributes_sha256===row.attributes_sha256,
      identical_canonical_attributes:baseline.canonical_attributes_sha256===row.canonical_attributes_sha256};
  });
  const controlReplay=control.rows.map(row=>{
    const previous=validation.cap.rows.find(other=>other.id===row.id);
    return {id:row.id,identical_output_and_attributes:row.output_sha256===previous.output_sha256 &&
      row.attributes_sha256===previous.attributes_sha256};
  });
  postHocFallbackValidation={control,experimental,score_gain_vs_same_binary_cap:experimental.summary.score-control.summary.score,
    rows,changed_rows:rows.filter(row=>!row.identical_output_and_attributes),
    identical_output_and_attributes:rows.filter(row=>row.identical_output_and_attributes).length,
    identical_canonical_attributes:rows.filter(row=>row.identical_canonical_attributes).length,
    topology_fallback_proposals:experimental.rows.reduce((n,row)=>n+row.topology_fallback_proposals,0),
    topology_fallback_assets:experimental.rows.filter(row=>row.topology_fallback_proposals>0).length,
    configured_gate_checks_per_kind:experimental.rows.length*(experimental.config.levels-1),
    candidate_evaluations:{control:control.rows.reduce((n,row)=>n+(row.candidate_evaluations??0),0),
      experimental:experimental.rows.reduce((n,row)=>n+(row.candidate_evaluations??0),0)},
    control_replay_identical_to_prior_cap:controlReplay.filter(row=>row.identical_output_and_attributes).length,
    control_replay_rows:controlReplay};
  const denseAudits=[];
  for(const row of postHocFallbackValidation.changed_rows) {
    const file='dense-validation-'+row.id+'-topology-fallback-v2.json';
    let data;
    try {data=await read('research/area-v3/'+file);}
    catch(error) {if(error.code==='ENOENT') {denseAudits.push({id:row.id,file,pending:true});continue;}throw error;}
    assert.equal(data.id,row.id,'E validation dense audit asset mismatch');
    assert.equal(data.run_sha256,experimental.run_sha256,'E validation dense audit run mismatch');
    if(fallbackHatchDense) {
      assert.deepEqual(data.audit,fallbackHatchDense.audit,'E validation dense contract mismatch');
      assert.equal(data.binary_sha256,fallbackHatchDense.auditor_binary_sha256,'E validation dense auditor mismatch');
    }
    denseAudits.push({id:row.id,file,pending:false,passed:data.passed,audit:data.audit,
      auditor_binary_sha256:data.binary_sha256,lods:data.lods,seconds:data.seconds});
  }
  postHocFallbackValidation.dense_audits=denseAudits;
}

async function strictScenarios() {
  const paths=[];
  for(const sourceCap of [3,4])for(const mode of ['b','e'])
    paths.push({sourceCap,mode,path:`research/runs/area-v3-strict-source-${sourceCap}-progressive-2-cap-0.5-${mode}`});
  try {await Promise.all(paths.map(item=>read(join(item.path,'summary.json'))));}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  const runs=await Promise.all(paths.map(async item=>({...item,run:await openRun(item.path)})));
  const pairs=[];
  for(const sourceCap of [3,4]) {
    const b=runs.find(item=>item.sourceCap===sourceCap && item.mode==='b').run;
    const e=runs.find(item=>item.sourceCap===sourceCap && item.mode==='e').run;
    comparableTopologyFallback(b,e);
    for(const run of [b,e]) {
      assert.equal(run.split,'development','strict scenario uses the wrong split');
      assert.equal(run.summary.expected,8,'strict scenario must use eight frozen assets');
      assert.ok(run.summary.complete,'strict scenario batch is incomplete');
      assert.equal(run.binary_sha256,postHocFallbackPilot.run.binary_sha256,'strict scenario uses another executable');
      assert.equal(run.manifest_sha256,cap.manifest_sha256,'strict scenario uses another pilot manifest');
      assert.equal(run.protocol_sha256,cap.protocol_sha256,'strict scenario uses another protocol');
      assert.equal(run.camera_sha256,cap.camera_sha256,'strict scenario uses other cameras');
      assert.equal(run.config.max_lod0_delta_px,sourceCap);
      assert.deepEqual(run.config.transition,[[0,2],[1,2]]);
      assert.equal(run.config.max_changed_area,0.5);
      assert.equal(run.config.candidate_budget,8);
      assert.equal(run.config.research.output,'rebuild');
      assert.equal(sameAssets(run),sameAssets(cap),'strict scenario changes pilot assets');
      const original=structuredClone(run===b?postHocFallbackPilot.control.config:postHocFallbackPilot.run.config);
      original.max_lod0_delta_px=sourceCap;original.transition=[[0,2],[1,2]];
      assert.deepEqual(run.config,original,'strict scenario changes more than the pixel contract');
    }
    const reference=b.rows.find(row=>row.source_limits.length===7);
    assert.ok(reference,'strict scenario has no audited LOD chain');
    for(const row of [...b.rows,...e.rows].filter(row=>row.source_limits.length===7)) {
      assert.deepEqual(row.source_limits,reference.source_limits,'strict source limit schedule differs by asset');
      assert.deepEqual(row.adjacent_limits,reference.adjacent_limits,'strict adjacent limit schedule differs by asset');
    }
    assert.ok(reference.adjacent_limits.every(value=>Math.abs(value-2)<1e-9),'strict transition schedule is not 2 px');
    assert.ok(reference.source_limits.every(value=>value<=sourceCap+1e-9),'strict source limit exceeds configured cap');
    const perAsset=e.rows.map(experimental=>{
      const control=b.rows.find(row=>row.id===experimental.id);
      return {id:experimental.id,category:experimental.category,control,experimental,
        identical_output_and_attributes:control.output_sha256===experimental.output_sha256 &&
          control.attributes_sha256===experimental.attributes_sha256};
    });
    pairs.push({source_cap_px:sourceCap,adjacent_cap_px:2,control:b,experimental:e,
      source_limits_px:reference.source_limits,adjacent_limits_px:reference.adjacent_limits,
      score_gain_points:e.summary.score-b.summary.score,
      per_asset:perAsset,changed_rows:perAsset.filter(row=>!row.identical_output_and_attributes),
      topology_fallback_proposals:e.rows.reduce((n,row)=>n+row.topology_fallback_proposals,0),
      configured_audits_pass:b.all_delivered_audits_pass && e.all_delivered_audits_pass});
  }
  const shelves=[];
  for(const pair of pairs)for(const [mode,run] of [['B',pair.control],['E',pair.experimental]]) {
    const id='ph_painted_wooden_shelves',raw=await read(join(run.path,'rows',id+'.json'));
    const row=run.rows.find(item=>item.id===id),lod6=raw.result.lods[6],lod7=raw.result.lods[7];
    shelves.push({source_cap_px:pair.source_cap_px,mode,run_sha256:run.run_sha256,
      final_triangles:row.final_triangles,lod6_triangles:lod6.triangles,lod7_triangles:lod7.triangles,
      lod6_source_px:lod6.source.error_px,lod6_source_limit_px:lod6.source_limit,
      lod6_adjacent_px:lod6.adjacent.error_px,lod6_adjacent_limit_px:lod6.transition_limit,
      source_audit_rejections:row.audit_rejections.source,adjacent_audit_rejections:row.audit_rejections.adjacent,
      area_only_source_audit_rejections:row.area_only_audit_rejections.source,
      area_only_adjacent_audit_rejections:row.area_only_audit_rejections.adjacent});
  }
  const crossCap={};
  const crossCapExport={};
  const withinCapExport={};
  for(const pair of pairs) {
    const matches=[];
    for(const item of pair.per_asset) {
      const hashes=await Promise.all(['chain.bin','chain.gltf'].flatMap(file=>[
        sha256(join(pair.control.path,'meshes',item.id,file)),
        sha256(join(pair.experimental.path,'meshes',item.id,file))]));
      const byteIdentical=hashes[0]===hashes[1] && hashes[2]===hashes[3];
      if(item.identical_output_and_attributes)assert.ok(byteIdentical,'same strict B/E output hashes have different exports');
      matches.push({id:item.id,byte_identical:byteIdentical});
    }
    withinCapExport[pair.source_cap_px]=matches;
  }
  for(const [mode,key] of [['B','control'],['E','experimental']]) {
    const lower=pairs[0][key],higher=pairs[1][key];
    crossCap[mode]=lower.rows.filter(row=>{
      const other=higher.rows.find(item=>item.id===row.id);
      return row.output_sha256===other.output_sha256 && row.attributes_sha256===other.attributes_sha256;
    }).length;
    let matchingExports=0;
    for(const row of lower.rows) {
      const hashes=await Promise.all(['chain.bin','chain.gltf'].flatMap(file=>[
        sha256(join(lower.path,'meshes',row.id,file)),sha256(join(higher.path,'meshes',row.id,file))]));
      if(hashes[0]===hashes[1] && hashes[2]===hashes[3])matchingExports++;
    }
    crossCapExport[mode]=matchingExports;
  }
  const denseE={};
  for(const [key,label,id] of [
    ['tree','Quiver tree','ph_dead_quiver_trunk'],
    ['grass','Bermuda grass','ph_grass_bermuda_01'],
    ['stool','Metal stool','ph_metal_stool_02'],
    ['moon-rock','Moon rock','ph_moon_rock_02'],
    ['shelves','Painted shelves','ph_painted_wooden_shelves'],
    ['rock-face','Rock face','ph_rock_face_02'],
    ['bell','Bell X-1','si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652'],
    ['hatch','Apollo hatch','si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7']]) {
    const file=`dense-strict-source-3-progressive-2-cap-0.5-${key}-e.json`;
    let data;
    try {data=await read('research/area-v3/'+file);}
    catch(error) {if(error.code==='ENOENT') {denseE[key]={id,label,file,pending:true};continue;}throw error;}
    assert.equal(data.id,id,'strict dense audit asset mismatch');
    assert.equal(data.run_sha256,pairs[0].experimental.run_sha256,'strict dense audit run mismatch');
    assert.equal(data.audit.max_changed_area,0.5);
    assert.equal(data.audit.seed,0xDA7A2026);
    denseE[key]={id,label,file,pending:false,passed:data.passed,audit:data.audit,lods:data.lods,
      auditor_binary_sha256:data.binary_sha256,seconds:data.seconds};
  }
  for(const item of Object.values(denseE).filter(value=>!value.pending))if(!denseE.tree.pending) {
    assert.deepEqual(item.audit,denseE.tree.audit,'strict dense audits use different contracts');
    assert.equal(item.auditor_binary_sha256,denseE.tree.auditor_binary_sha256,'strict dense audits use different binaries');
  }
  const extraDense=[];
  for(const [sourceCap,mode,label,id,suffix,seed] of [
    [3,'B','Bell X-1','si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652','bell-b',0xDA7A2026],
    [3,'B','Apollo hatch','si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7','hatch-b',0xDA7A2026],
    [4,'B','Bell X-1','si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652','bell-b',0xDA7A2026],
    [4,'B','Apollo hatch','si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7','hatch-b',0xDA7A2026],
    [4,'E','Quiver tree','ph_dead_quiver_trunk','tree-e',0xDA7A2026],
    [4,'E','Bermuda grass','ph_grass_bermuda_01','grass-e',0xDA7A2026],
    [4,'E','Metal stool','ph_metal_stool_02','stool-e',0xDA7A2026],
    [4,'E','Moon rock','ph_moon_rock_02','moon-rock-e',0xDA7A2026],
    [4,'E','Painted shelves','ph_painted_wooden_shelves','shelves-e',0xDA7A2026],
    [4,'E','Rock face','ph_rock_face_02','rock-face-e',0xDA7A2026],
    [4,'E','Bell X-1','si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652','bell-e',0xDA7A2026],
    [4,'E','Apollo hatch','si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7','hatch-e',0xDA7A2026],
    [3,'E','Bell X-1, second seed','si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652','bell-e-seed-2027',0xDA7A2027],
    [3,'E','Moon rock, second seed','ph_moon_rock_02','moon-rock-e-seed-2027',0xDA7A2027]]) {
    const file=`dense-strict-source-${sourceCap}-progressive-2-cap-0.5-${suffix}.json`;
    let data;
    try {data=await read('research/area-v3/'+file);}
    catch(error) {if(error.code==='ENOENT')continue;throw error;}
    const run=pairs.find(pair=>pair.source_cap_px===sourceCap)[mode==='B'?'control':'experimental'];
    assert.equal(data.id,id,'additional strict dense asset mismatch');
    assert.equal(data.run_sha256,run.run_sha256,'additional strict dense run mismatch');
    assert.equal(data.audit.seed,seed,'additional strict dense seed mismatch');
    assert.equal(data.audit.max_changed_area,0.5);
    if(!denseE.tree.pending)assert.equal(data.binary_sha256,denseE.tree.auditor_binary_sha256,
      'additional strict dense auditor mismatch');
    extraDense.push({source_cap_px:sourceCap,mode,label,id,file,passed:data.passed,audit:data.audit,
      lods:data.lods,auditor_binary_sha256:data.binary_sha256,seconds:data.seconds});
  }
  const treeId='ph_dead_quiver_trunk',treeHashes={};
  for(const file of ['chain.bin','chain.gltf']) {
    treeHashes[file]={b:await sha256(join(pairs[0].control.path,'meshes',treeId,file)),
      e:await sha256(join(pairs[0].experimental.path,'meshes',treeId,file))};
  }
  const treeByteIdentical=Object.values(treeHashes).every(item=>item.b===item.e);
  assert.ok(treeByteIdentical,'strict 3px tree B/E export differs');
  const directE4=extraDense.filter(item=>item.source_cap_px===4 && item.mode==='E' &&
    item.audit.seed===0xDA7A2026);
  let identicalMeasurements=0;
  for(const item of Object.values(denseE).filter(value=>!value.pending)) {
    const other=directE4.find(candidate=>candidate.id===item.id);
    if(!other)continue;
    assert.equal(item.lods.length,other.lods.length);
    for(let i=0;i<item.lods.length;i++) {
      assert.equal(item.lods[i].triangles,other.lods[i].triangles);
      assert.deepEqual(item.lods[i].source,other.lods[i].source);
      assert.deepEqual(item.lods[i].adjacent,other.lods[i].adjacent);
    }
    identicalMeasurements++;
  }
  const firstRotationE3=Object.values(denseE).length===8 && Object.values(denseE).every(item=>!item.pending && item.passed);
  const firstRotationE4=directE4.length===8 && directE4.every(item=>item.passed);
  const bEvidence={};
  for(const sourceCap of [3,4]) {
    const direct=extraDense.filter(item=>item.source_cap_px===sourceCap && item.mode==='B' &&
      ['si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652',
        'si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7'].includes(item.id));
    const inherited=withinCapExport[sourceCap].filter(item=>item.byte_identical &&
      (sourceCap===3?Object.values(denseE).find(audit=>audit.id===item.id)?.passed:
        directE4.find(audit=>audit.id===item.id)?.passed));
    bEvidence[sourceCap]={all8_pass:(sourceCap===3?firstRotationE3:firstRotationE4) &&
      inherited.length===6 && direct.length===2 && direct.every(item=>item.passed &&
        item.audit.seed===0xDA7A2026),
      identity_derived_assets:inherited.map(item=>item.id),direct_assets:direct.map(item=>item.id)};
  }
  return {purpose:'User-directed strict pixel-contract scenarios; separate from frozen A/B/C and post-hoc E.',pairs,shelves,
    cross_cap_identical_output_and_attributes:crossCap,
    cross_cap_byte_identical_exports:crossCapExport,dense_3px_e:denseE,extra_dense:extraDense,
    within_cap_byte_identical_exports:withinCapExport,
    strict_tree_b_e_export:{byte_identical:treeByteIdentical,hashes:treeHashes},
    first_rotation_tail:{e_3px_direct_all8_pass:firstRotationE3,e_4px_direct_all8_pass:firstRotationE4,
      b_3px:bEvidence[3],b_4px:bEvidence[4],e_3px_4px_identical_measured_assets:identicalMeasurements}};
}
const strict=await strictScenarios();

const newUnreduced=cap.rows.filter(row=>row.unreduced && !uncapped.rows.find(other=>other.id===row.id)?.unreduced).map(row=>row.id);
const tree=cap.rows.find(row=>row.id==='ph_dead_quiver_trunk');
const bellId='si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652';
const bellCap=cap.rows.find(row=>row.id===bellId),bellAdaptive=adaptive.rows.find(row=>row.id===bellId);
const generationSeconds=run=>run.rows.reduce((sum,row)=>sum+(row.generation_seconds??0),0);
const scoreCost=uncapped.summary.score!==null && cap.summary.score!==null?uncapped.summary.score-cap.summary.score:null;
const adaptiveGain=cap.summary.score!==null && adaptive.summary.score!==null?adaptive.summary.score-cap.summary.score:null;
const presetPilot=uncapped.summary.complete && cap.all_delivered_audits_pass && tree?.final_ratio<0.01 && newUnreduced.length===0 && scoreCost<=5;
const adaptivePilot=cap.all_delivered_audits_pass && adaptive.all_delivered_audits_pass && adaptiveGain>=1;
const validationScoreCost=validation && validation.uncapped.summary.score!==null && validation.cap.summary.score!==null?
  validation.uncapped.summary.score-validation.cap.summary.score:null;
const capValidation=validation?validation.uncapped.all_delivered_audits_pass && validation.cap.all_delivered_audits_pass &&
  validationScoreCost<=5:null;
const adaptiveValidation=validation?.adaptive?validation.cap.all_delivered_audits_pass && validation.adaptive.all_delivered_audits_pass &&
  validation.adaptive.summary.score>=validation.cap.summary.score && validation.adaptive.failures.length<=validation.cap.failures.length:null;
const report={version:2,scope:'Opaque geometry; fixed cameras and supersampling. Dense tail audit is reported separately.',
  pilot:{uncapped,cap,adaptive},validation,validation_change:validationChange,
  compatibility,final_binary_replay:finalReplay,
  dense_tail:{rotated:dense,default_seed:denseDefault},exploratory,post_hoc_topology:postHoc,
  post_hoc_topology_dense:dDense,
  post_hoc_topology_fallback:{pilot:postHocFallbackPilot,validation:postHocFallbackValidation,
    dense:{hatch:fallbackHatchDense,bell:fallbackBellDense,tree:fallbackTreeDense},
    inherited_tree_dense_failure:postHocFallbackPilot?.tree_byte_identical===true && dense?.passed===false},
  strict_scenarios:strict,
  decisions:{score_cost_points:scoreCost,new_unreduced_assets:newUnreduced,tree_final_ratio:tree?.final_ratio??null,
    cap_pilot_qualifies:presetPilot,validation_score_cost_points:validationScoreCost,cap_validation_qualifies:capValidation,
    cap_dense_qualifies:dense && denseDefault?dense.passed && denseDefault.passed:null,
    cap_promotion_ready:presetPilot && capValidation===true && dense?.passed===true && denseDefault?.passed===true,
    adaptive_gain_points:adaptiveGain,adaptive_pilot_qualifies:adaptivePilot,
    adaptive_validation_qualifies:adaptiveValidation,adaptive_selected:adaptiveValidation===null?null:adaptivePilot && adaptiveValidation,
    topology_fallback_validation_gain_points:postHocFallbackValidation?.score_gain_vs_same_binary_cap??null,
    topology_fallback_changed_validation_dense_pass:postHocFallbackValidation?
      postHocFallbackValidation.dense_audits.some(item=>item.pending)?null:
      postHocFallbackValidation.dense_audits.every(item=>item.passed):null,
    topology_fallback_opt_in_only:true}};

const lines=['# Coverage-area quality measurements','',
  '[Interactive development board](board/index.html) · [Raw derived analysis](analysis.json)','',
  'Generated from complete benchmark rows by make-report.mjs. Frozen A/B/C use the same executable, pilot assets, cameras, supersampling and eight-proposal budget. Source and adjacent coverage-area limits are checked separately from pixel distance. Scores across area limits describe the quality–triangle tradeoff; within frozen A/B/C, only B versus C is an algorithm comparison. Post-hoc E uses a later executable and can add bounded proposals; its same-binary B controls are reported separately.','',
  'The archived [Round 4 v1 tree result](../round4/REPORT.md) motivated this experiment; its score is not a v3 baseline. The fresh v3 A/B/C runs below provide the measured comparison.','',
  '| Variant | Complete | SCORE | Final retained, category mean | Final source area max | Final/LOD0 tree tris | Fallbacks | Failed | Generation s | Process RSS high-water MiB |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'];
for(const run of [uncapped,cap,adaptive]) {
  const treeRow=run.rows.find(row=>row.id==='ph_dead_quiver_trunk');
  const area=Math.max(0,...run.rows.map(row=>row.final_source_area??0));
  const seconds=generationSeconds(run);
  const rss=Math.max(0,...run.rows.map(row=>row.peak_rss_kib??0))/1024;
  lines.push('| '+run.name+' | '+run.summary.completed+'/'+run.summary.expected+' | '+fixed(run.summary.score)+' | '+pct(run.category_balanced_final_ratio)+' | '+pct(area)+' | '+
    (treeRow?.final_triangles??'—')+'/'+(treeRow?.source_triangles??'—')+' | '+run.summary.fallbacks+' | '+run.failures.length+' | '+fixed(seconds)+' | '+fixed(rss)+' |');
}
lines.push('','## Per-asset measurements','',
  '| Variant / asset | Tris source→final | Final retained | Worst source area | Worst adjacent area | Resident KiB | Generation s | Process RSS high-water MiB | State |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---|');
for(const run of [uncapped,cap,adaptive])for(const row of run.rows) {
  const state=row.failed?'FAILED'+(row.failure?': '+row.failure:''):row.violations.length?'AUDIT FAIL':row.fallback?'fallback':'pass';
  lines.push('| '+run.name+' / '+row.id+' | '+(row.source_triangles??'—')+'→'+(row.final_triangles??'—')+' | '+pct(row.final_ratio)+
    ' | '+location(row.source_area)+' | '+location(row.adjacent_area)+' | '+fixed(row.resident_bytes===null?null:row.resident_bytes/1024)+
    ' | '+fixed(row.generation_seconds)+' | '+fixed(row.peak_rss_kib===null?null:row.peak_rss_kib/1024)+' | '+state+' |');
}
lines.push('','## Pilot decision','',
  '- Area preset pilot qualification: '+(presetPilot?'pass':'fail or incomplete')+'. SCORE cost '+fixed(scoreCost)+' points; new unreduced assets: '+(newUnreduced.join(', ')||'none')+'; tree final retention: '+pct(tree?.final_ratio??null)+'.',
  '- Adaptive pilot qualification at the 0.5 limit: '+(adaptivePilot?'pass':'fail or incomplete')+'. SCORE gain '+fixed(adaptiveGain)+' points.',
  '- Adaptive Bell X-1 chain retention changed from '+pct(bellCap?.chain_ratio??null)+' to '+pct(bellAdaptive?.chain_ratio??null)+
    '; adaptive generation took '+fixed(generationSeconds(adaptive)-generationSeconds(cap))+' more seconds across the pilot.');
if(postHoc) {
  const run=postHoc.run;
  lines.push('','## Post-hoc topology-relaxed development pilot','',
    'This objective-changing test followed the frozen A/B/C comparison. It uses the same eight development assets, 0.5 area limit, cameras, sampling, work budget, executable and protocol as B. It is exploratory; its topology behavior and generalization need separate review. The held-out split was not used. [Raw D summary](../runs/area-v3-cap-0.5-topology-relaxed/summary.json).','',
    'D completed '+run.summary.completed+'/'+run.summary.expected+' assets, with '+run.failures.length+' failures and '+run.summary.fallbacks+' fallbacks. SCORE '+fixed(run.summary.score)+
    ' versus B '+fixed(cap.summary.score)+' (change '+fixed(postHoc.score_gain_vs_cap)+' points); generation time '+fixed(generationSeconds(run))+
    ' s versus B '+fixed(generationSeconds(cap))+' s. All configured audits pass: '+(run.all_delivered_audits_pass?'yes':'no')+'. '+
    'Category-balanced final retained ratio is '+pct(run.category_balanced_final_ratio)+' versus B '+pct(cap.category_balanced_final_ratio)+'. '+
    (postHoc.score_gain_vs_cap<0?'The aggregate SCORE regression rules out a global objective switch on this pilot.':'This post-hoc gain requires independent validation before any global objective switch.'),'',
    '| Asset | Final tris B→D | Chain retained B→D | Final source area B→D | Resident KiB B→D | Generation s B→D | State D |',
    '|---|---:|---:|---:|---:|---:|---|');
  for(const item of postHoc.per_asset) {
    const a=item.cap,d=item.topology_relaxed;
    lines.push('| '+item.id+' | '+(a.final_triangles??'—')+'→'+(d.final_triangles??'—')+' | '+pct(a.chain_ratio)+'→'+pct(d.chain_ratio)+
      ' | '+pct(a.final_source_area)+'→'+pct(d.final_source_area)+' | '+fixed(a.resident_bytes===null?null:a.resident_bytes/1024)+'→'+
      fixed(d.resident_bytes===null?null:d.resident_bytes/1024)+' | '+fixed(a.generation_seconds)+'→'+fixed(d.generation_seconds)+
      ' | '+(d.failed?'FAILED':d.violations.length?'AUDIT FAIL':d.fallback?'fallback':'pass')+' |');
  }
  if(hatchDense||bellDense||rockDense||treeDense) {
    lines.push('','Independent 642+64-view rotated dense tail checks at 8× with refinement to 32×:','',
      '| D asset / raw audit | Final tris | LOD7 source area / cap | LOD7 adjacent area / cap | Six tail gates |',
      '|---|---:|---:|---:|---|');
    for(const [label,file,item] of [
      ['Apollo hatch','dense-hatch-topology-relaxed.json',hatchDense],
      ['Bell X-1','dense-bell-topology-relaxed.json',bellDense],
      ['Rock face','dense-rock-topology-relaxed.json',rockDense],
      ['Quiver tree','dense-tree-topology-relaxed.json',treeDense]])if(item) {
      const last=item.lods.at(-1);
      lines.push('| ['+label+']('+file+') | '+last.triangles+' | '+pct(last.source.changed_area)+' / '+pct(item.audit.max_changed_area)+
        ' | '+pct(last.adjacent.changed_area)+' / '+pct(item.audit.max_changed_area)+' | '+(item.passed?'pass':'FAIL')+' |');
    }
    const auditor=hatchDense??bellDense??rockDense??treeDense;
    if(rockDense&&!rockDense.passed)lines.push('','The rock face fails LOD7 source and adjacent area at '+
      pct(rockDense.lods.at(-1).source.changed_area)+' and '+pct(rockDense.lods.at(-1).adjacent.changed_area)+
      '; both pixel-distance gates pass.');
    if(treeDense&&!treeDense.passed)lines.push('','The quiver tree fails LOD7 source area at '+
      pct(treeDense.lods.at(-1).source.changed_area)+' (view '+treeDense.lods.at(-1).source.changed_area_worst_view+
      '); its pixel-distance gate passes.');
    lines.push('','These object-specific checks support hatch and Bell X-1 headroom, but do not justify global topology relaxation. D run SHA-256: '+run.run_sha256+
      '; D generator binary SHA-256: '+run.binary_sha256+'; dense auditor binary SHA-256: '+auditor.auditor_binary_sha256+'.');
  }
}
if(postHocFallbackPilot) {
  const pilot=postHocFallbackPilot,run=pilot.run,control=pilot.control;
  lines.push('','## Post-hoc conditional topology fallback E','',
    'E followed the frozen A/B/C pilot and the broader topology-relaxed D probe. It keeps B’s quadric objective and 0.5 area limit, but tries at most one topology-relaxed proposal per LOD when a quadric reduction stops more than four times above its requested triangles with link-condition rejections. Those proposals are extra work beyond the nominal eight-proposal budget. '+
    (pilot.strict_same_binary_comparison?
      'The archived [post-hoc B pilot control](../runs/area-v3-cap-0.5-fallback-v2-b/summary.json) uses E’s corrected executable; '+pilot.frozen_control_replay_identical+'/8 output and attribute hashes match frozen B.':
      'E uses a later executable than frozen pilot B, so its pilot SCORE difference is descriptive pending the same-binary B control.')+
    (postHocFallbackValidation?' The same-binary validation follow-up appears below.':'')+
    ' [E config](configs/cap-0.5-topology-fallback.json) · [corrected v2 E pilot](../runs/area-v3-cap-0.5-fallback-v2-e/summary.json).','',
    'This is a bounded extra-work comparison, not evidence of superiority at equal reducer work. E completed '+run.summary.completed+'/'+run.summary.expected+' development assets, with '+run.failures.length+' failures and '+run.summary.fallbacks+
    ' fallbacks. SCORE '+precise(run.summary.score)+' versus '+(pilot.strict_same_binary_comparison?'same-binary B control ':'frozen B ')+
    precise(control.summary.score)+' (change '+precise(pilot.score_gain_vs_control)+
    ' points). All configured audits pass: '+(run.all_delivered_audits_pass?'yes':'no')+'. It made '+pilot.topology_fallback_proposals+
    ' extra topology proposals; total candidate evaluations were '+pilot.candidate_evaluations.control+' in B and '+
    pilot.candidate_evaluations.experimental+' in E. '+pilot.identical_output_and_attributes+'/'+run.summary.expected+
    ' output and attribute hashes match B, with '+pilot.identical_canonical_attributes+'/'+run.summary.expected+
    ' canonical input attribute hashes matching. Category-balanced final retention is '+pct(run.category_balanced_final_ratio)+
    ' versus B '+pct(control.category_balanced_final_ratio)+'. Generation time is '+fixed(generationSeconds(run))+' s versus B '+
    fixed(generationSeconds(control))+' s; process RSS high-water is '+
    fixed(Math.max(...run.rows.map(row=>row.peak_rss_kib??0))/1024)+' MiB versus B '+
    fixed(Math.max(...control.rows.map(row=>row.peak_rss_kib??0))/1024)+' MiB.','',
    '| Changed development asset | Final tris B→E | Chain retained B→E | Final source area B→E | Extra proposals E |',
    '|---|---:|---:|---:|---:|');
  for(const row of pilot.changed_rows)lines.push('| '+row.id+' | '+row.control.final_triangles+'→'+row.experimental.final_triangles+
    ' | '+pct(row.control.chain_ratio)+'→'+pct(row.experimental.chain_ratio)+' | '+
    pct(row.control.final_source_area)+'→'+pct(row.experimental.final_source_area)+' | '+
    row.experimental.topology_fallback_proposals+' |');
  if(fallbackHatchDense||fallbackBellDense||fallbackTreeDense) {
    lines.push('','Independent 642+64-view rotated dense tail checks at 8× with refinement to 32×:','',
      '| E asset / raw audit | Final tris | LOD7 source area / cap | LOD7 adjacent area / cap | Six tail gates |',
      '|---|---:|---:|---:|---|');
    for(const [label,file,item] of [
      ['Apollo hatch','dense-hatch-topology-fallback-v2.json',fallbackHatchDense],
      ['Bell X-1','dense-bell-topology-fallback-v2.json',fallbackBellDense],
      ['Quiver tree','dense-tree-topology-fallback-v2.json',fallbackTreeDense]])if(item) {
      const last=item.lods.at(-1);
      lines.push('| ['+label+']('+file+') | '+last.triangles+' | '+pct(last.source.changed_area)+' / '+pct(item.audit.max_changed_area)+
        ' | '+pct(last.adjacent.changed_area)+' / '+pct(item.audit.max_changed_area)+' | '+(item.passed?'pass':'FAIL')+' |');
    }
  }
  lines.push('','The corrected v2 B and E outputs reproduce their prior eight-asset exports: '+
    pilot.frozen_control_replay_identical+'/8 B and '+pilot.historical_e.identical_output_and_attributes+'/8 E output and attribute hashes match. '+
    'The pre-fix E run is [historical evidence](../runs/area-v3-cap-0.5-topology-fallback/summary.json), not the scored comparison. '+
    'Corrected E’s tree `chain.bin` and `chain.gltf` are byte-identical to frozen B’s: '+
    (pilot.tree_byte_identical?'yes':'NO')+'. '+
    (fallbackTreeDense?'The direct corrected E rotated tree audit '+(fallbackTreeDense.passed?'passes':'fails')+
      ' LOD7 source area at '+pct(fallbackTreeDense.lods.at(-1).source.changed_area)+', matching frozen B’s rotated tree result.':
      'A direct corrected E dense tree audit is pending; the byte-identical frozen B tree fails the rotated area check at '+
      pct(dense?.lods.at(-1)?.source.changed_area??null)+'.')+
    ' '+(fallbackHatchDense&&fallbackBellDense?
      'Passing corrected hatch and Bell checks does not resolve this tree miss.':
      'The historical hatch and Bell checks passed on byte-identical exports; direct corrected audits are pending.')+' '+
    'The fallback relaxes the link-condition topology restriction for selected proposals; it does not guarantee manifold topology or exclude new intersections. Keep E opt-in and inspect each chosen export. '+
    'E pilot run SHA-256: '+run.run_sha256+'; E generator binary SHA-256: '+run.binary_sha256+
    (fallbackHatchDense?'; E dense auditor binary SHA-256: '+fallbackHatchDense.auditor_binary_sha256:'')+'.');
  if(postHocFallbackValidation) {
    const item=postHocFallbackValidation,control=item.control,experimental=item.experimental;
    lines.push('','### Same-binary E validation follow-up','',
      'This post-hoc 20-asset comparison uses the corrected v2 executable, validation manifest, cameras and 0.5 cap for its B control and E run. The only setting change is `research.topology_fallback`, which permits at most one extra reducer call per LOD. It is a bounded extra-work opt-in comparison, not equal-work optimizer superiority, and is separate from the frozen A/B validation. [B control](../runs/area-v3-validation-cap-0.5-fallback-v2-b/summary.json) · [E validation](../runs/area-v3-validation-cap-0.5-fallback-v2-e/summary.json).','',
      '| Variant | Complete | SCORE | Final retained, category mean | Worst final source area | Fallbacks | Failed | Configured audits pass | Generation s | Process RSS high-water MiB |',
      '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|');
    for(const current of [control,experimental])lines.push('| '+current.name+' | '+current.summary.completed+'/'+current.summary.expected+
      ' | '+precise(current.summary.score)+' | '+pct(current.category_balanced_final_ratio)+' | '+
      pct(Math.max(0,...current.rows.map(row=>row.final_source_area??0)))+' | '+current.summary.fallbacks+
      ' | '+current.failures.length+' | '+(current.all_delivered_audits_pass?'yes':'no')+' | '+
      fixed(generationSeconds(current))+' | '+
      fixed(Math.max(...current.rows.map(row=>row.peak_rss_kib??0))/1024)+' |');
    lines.push('','E versus same-binary B control: '+precise(item.score_gain_vs_same_binary_cap)+' SCORE points, '+
      item.topology_fallback_proposals+' extra topology proposals across '+item.topology_fallback_assets+' assets and '+
      item.candidate_evaluations.control+'→'+
      item.candidate_evaluations.experimental+' total candidate evaluations. '+item.identical_output_and_attributes+
      '/'+experimental.summary.expected+' output and attribute hashes remain identical; '+item.identical_canonical_attributes+
      '/'+experimental.summary.expected+' canonical input attribute hashes match. The new B control reproduces '+
      item.control_replay_identical_to_prior_cap+'/'+control.summary.expected+
      ' output and attribute hashes from the earlier capped validation binary. Both variants pass all '+
      item.configured_gate_checks_per_kind+' source and '+item.configured_gate_checks_per_kind+
      ' adjacent configured LOD audits.','',
      '| Changed validation asset | Final tris control→E | Chain retained control→E | Final source area control→E | Extra proposals E |',
      '|---|---:|---:|---:|---:|');
    for(const row of item.changed_rows)lines.push('| '+row.id+' | '+row.control.final_triangles+'→'+row.experimental.final_triangles+
      ' | '+pct(row.control.chain_ratio)+'→'+pct(row.experimental.chain_ratio)+' | '+
      pct(row.control.final_source_area)+'→'+pct(row.experimental.final_source_area)+' | '+
      row.experimental.topology_fallback_proposals+' |');
    if(!item.changed_rows.length)lines.push('| None | — | — | — | — |');
    if(item.changed_rows.length) {
      lines.push('','Independent rotated dense tail audits of changed E validation exports:','',
        '| Asset / raw audit | Final tris | LOD7 source area / cap | LOD7 adjacent area / cap | Source px / limit | Adjacent px / limit | Six tail gates |',
        '|---|---:|---:|---:|---:|---:|---|');
      for(const audit of item.dense_audits) {
        const row=item.changed_rows.find(other=>other.id===audit.id);
        if(audit.pending)lines.push('| '+audit.id+' (pending) | '+row.experimental.final_triangles+' | — | — | — | — | pending |');
        else {
          const last=audit.lods.at(-1);
          lines.push('| ['+audit.id+']('+audit.file+') | '+last.triangles+' | '+
            pct(last.source.changed_area)+' / '+pct(audit.audit.max_changed_area)+' | '+
            pct(last.adjacent.changed_area)+' / '+pct(audit.audit.max_changed_area)+' | '+
            fixed(last.source.error_px)+' / '+fixed(last.source_limit)+' | '+
            fixed(last.adjacent.error_px)+' / '+fixed(last.transition_limit)+' | '+
            (audit.passed?'pass':'FAIL')+' |');
        }
      }
      lines.push('','Changed-export dense checks: '+item.dense_audits.filter(audit=>audit.passed).length+'/'+
        item.changed_rows.length+' pass'+(item.dense_audits.some(audit=>audit.pending)?'; pending audits remain.':'.'));
    }
    lines.push('','This measured follow-up does not promote the original 8 px/variable-transition E scenario globally: its unchanged tree still fails the independent rotated dense area check.');
  }
}
if(strict) {
  lines.push('','## Strict pixel-contract scenarios','',
    'This user-directed development follow-up uses the corrected v2 executable, frozen eight assets, cameras, 0.5 changed-area cap, eight-proposal nominal budget and hybrid rebuild chain. It fixes the adjacent/progressive pixel limit at 2 px, then evaluates source caps of 3 px and 4 px. Each source cap has a same-binary quadric B control and conditional-fallback E run. E may add one reducer call per LOD. SCORE comparisons below are only B versus E within the same pixel contract; scores from different source caps are not pooled or ranked. [Separate all-asset strict board](strict-board/index.html).','',
    'All four configs force research rebuild and triangle-first chain selection. Metadata retains the default `triangle_overhead_bps=500`, but this research selection path does not apply a 5% triangle allowance.');
  for(const pair of strict.pairs) {
    const b=pair.control,e=pair.experimental;
    lines.push('','### Source cap '+pair.source_cap_px+' px; adjacent cap 2 px','',
      'Actual LOD1–7 source limits: '+pair.source_limits_px.map(value=>fixed(value)).join(', ')+' px. '+
      'Adjacent limits: '+pair.adjacent_limits_px.map(value=>fixed(value)).join(', ')+' px. '+
      '[B config](configs/strict-source-'+pair.source_cap_px+'-progressive-2-cap-0.5-b.json) · '+
      '[E config](configs/strict-source-'+pair.source_cap_px+'-progressive-2-cap-0.5-e.json).','',
      '| Run | Complete | SCORE | Category mean final retention | Worst source area, any LOD | Worst adjacent area, any LOD | Fallbacks | Failed | Configured audits | Generation s | RSS high-water MiB |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|');
    for(const run of [b,e])lines.push('| ['+run.name+'](../runs/'+run.name+'/summary.json) | '+run.summary.completed+'/'+run.summary.expected+
      ' | '+precise(run.summary.score)+' | '+pct(run.category_balanced_final_ratio)+' | '+
      pct(Math.max(0,...run.rows.map(row=>row.source_area?.value??0)))+' | '+
      pct(Math.max(0,...run.rows.map(row=>row.adjacent_area?.value??0)))+' | '+
      run.summary.fallbacks+' | '+run.failures.length+' | '+(run.all_delivered_audits_pass?'pass':'FAIL')+' | '+
      fixed(generationSeconds(run))+' | '+fixed(Math.max(...run.rows.map(row=>row.peak_rss_kib??0))/1024)+' |');
    lines.push('','Within this '+pair.source_cap_px+' px contract, E minus B SCORE is '+precise(pair.score_gain_points)+
      ' points with '+pair.topology_fallback_proposals+' extra topology proposals. '+
      pair.changed_rows.length+'/'+e.summary.expected+' output/attribute hash pairs change. '+
      'Both configured audits pass: '+(pair.configured_audits_pass?'yes':'no')+'.','',
      '| Asset | Final tris B→E | Chain retained B→E | Final source area B→E | Source/adjacent audit rejects B→E | Extra E calls | State B/E |',
      '|---|---:|---:|---:|---:|---:|---|');
    for(const item of pair.per_asset) {
      const a=item.control,d=item.experimental;
      const state=row=>row.failed?'FAILED':row.violations.length?'AUDIT FAIL':row.fallback?'fallback':'pass';
      lines.push('| '+item.id+' | '+(a.final_triangles??'—')+'→'+(d.final_triangles??'—')+
        ' | '+pct(a.chain_ratio)+'→'+pct(d.chain_ratio)+' | '+
        pct(a.final_source_area)+'→'+pct(d.final_source_area)+' | '+
        a.audit_rejections.source+'/'+a.audit_rejections.adjacent+'→'+
        d.audit_rejections.source+'/'+d.audit_rejections.adjacent+' | '+
        d.topology_fallback_proposals+' | '+state(a)+'/'+state(d)+' |');
    }
  }
  const denseItems=Object.entries(strict.dense_3px_e).filter(([,item])=>!item.pending);
  if(denseItems.length) {
    const auditReference=denseItems[0][1].audit;
    const worstGate=(item,kind,metric)=>{
      let best=null;
      for(const lod of item.lods) {
        const value=lod[kind][metric];
        const limit=metric==='error_px'?lod[kind==='source'?'source_limit':'transition_limit']:lod.max_changed_area;
        if(!best || value/limit>best.value/best.limit)best={level:lod.level,value,limit};
      }
      return best;
    };
    lines.push('','### Independent rotated strict-tail checks','',
      'Selected 3 px E exports were checked on '+auditReference.orthographic+'+'+
      auditReference.perspective+' cameras at 8× with refinement to 32×, seed '+
      auditReference.seed+'. This is one independent rotation; passing it does not prove an all-view bound. '+
      'Each cell below is the tightest margin across LOD5–7, with its LOD.','',
      '| Asset / raw audit | Final tris | Source area / cap | Adjacent area / cap | Source px / limit | Adjacent px / limit | Six tail gates |',
      '|---|---:|---:|---:|---:|---:|---|');
    for(const [label,item] of denseItems) {
      const srcArea=worstGate(item,'source','changed_area'),adjArea=worstGate(item,'adjacent','changed_area');
      const srcPx=worstGate(item,'source','error_px'),adjPx=worstGate(item,'adjacent','error_px');
      lines.push('| ['+item.label+']('+item.file+') | '+item.lods.at(-1).triangles+' | '+
        pct(srcArea.value)+' / '+pct(srcArea.limit)+' (LOD '+srcArea.level+') | '+
        pct(adjArea.value)+' / '+pct(adjArea.limit)+' (LOD '+adjArea.level+') | '+
        fixed(srcPx.value)+' / '+fixed(srcPx.limit)+' (LOD '+srcPx.level+') | '+
        adjPx.value.toFixed(5)+' / '+fixed(adjPx.limit)+' (LOD '+adjPx.level+') | '+
        (item.passed?'pass':'FAIL')+' |');
    }
    const nearBell=strict.dense_3px_e.bell.pending?null:worstGate(strict.dense_3px_e.bell,'adjacent','error_px');
    if(nearBell)lines.push('','Bell X-1’s tightest adjacent-pixel margin is '+
      (nearBell.limit-nearBell.value).toFixed(5)+' px at LOD '+nearBell.level+
      '. A different rotation could still expose a failure.');
    if(strict.extra_dense.length) {
      lines.push('','Additional direct checks on B and 4 px E exports, plus the second Bell rotation:','',
        '| Contract / asset / raw audit | Final tris | Source px / limit | Adjacent px / limit | Six tail gates |',
        '|---|---:|---:|---:|---|');
      for(const item of strict.extra_dense) {
        const source=worstGate(item,'source','error_px'),adjacent=worstGate(item,'adjacent','error_px');
        lines.push('| ['+item.source_cap_px+' px '+item.mode+' '+item.label+']('+item.file+') | '+
          item.lods.at(-1).triangles+' | '+fixed(source.value)+' / '+fixed(source.limit)+' (LOD '+source.level+') | '+
          adjacent.value.toFixed(5)+' / '+fixed(adjacent.limit)+' (LOD '+adjacent.level+') | '+
          (item.passed?'pass':'FAIL')+' |');
      }
      const secondBell=strict.extra_dense.find(item=>item.file.endsWith('bell-e-seed-2027.json'));
      if(secondBell)lines.push('','The second Bell E seed '+secondBell.audit.seed+' '+
        (secondBell.passed?'passes':'FAILS')+' all six tail gates. Its smallest adjacent-pixel margin is '+
        (worstGate(secondBell,'adjacent','error_px').limit-
          worstGate(secondBell,'adjacent','error_px').value).toFixed(5)+
        ' px. Two finite rotations still do not establish all-view robustness.');
      const secondMoon=strict.extra_dense.find(item=>item.file.endsWith('moon-rock-e-seed-2027.json'));
      if(secondMoon)lines.push('','The second Moon rock E seed '+secondMoon.audit.seed+' '+
        (secondMoon.passed?'passes':'FAILS')+' all six tail gates; its smallest adjacent-pixel margin is '+
        (worstGate(secondMoon,'adjacent','error_px').limit-
          worstGate(secondMoon,'adjacent','error_px').value).toFixed(5)+' px.');
    }
    const tail=strict.first_rotation_tail;
    lines.push('','First rotated-seed coverage: direct 3 px E 8/8 '+(tail.e_3px_direct_all8_pass?'pass':'not fully passed')+
      ', direct 4 px E 8/8 '+(tail.e_4px_direct_all8_pass?'pass':'not fully passed')+
      '; 3 px B 8/8 '+(tail.b_3px.all8_pass?'pass':'not fully passed')+
      ' and 4 px B 8/8 '+(tail.b_4px.all8_pass?'pass':'not fully passed')+
      ' using six byte-identical B/E exports plus direct Bell and hatch audits per contract. '+
      'Each asset has six LOD5–7 source/adjacent gates.','',
      'The 4 px E exports are byte-identical to their 3 px counterparts and have no tighter source limit; their adjacent and area limits are unchanged. Their source/adjacent measurement objects are identical for '+
      tail.e_3px_4px_identical_measured_assets+'/8 directly audited assets; only the source limits differ. These are finite sampled camera sets, not all-view guarantees.');
  }
  const strictTree=strict.pairs[0].per_asset.find(item=>item.id==='ph_dead_quiver_trunk');
  if(strictTree&&strict.dense_3px_e.tree&&!strict.dense_3px_e.tree.pending)lines.push('',
    'The 3 px tree B and E exports are byte-identical at '+strictTree.control.final_triangles+
    ' final triangles, so topology fallback did not cause this tree change. Frozen 8 px/variable-transition B has '+
    tree.final_triangles+' final triangles and fails the rotated dense source-area cap at '+
    pct(dense.lods.at(-1).source.changed_area)+'. The strict 3 px tree passes that first rotation at '+
    pct(strict.dense_3px_e.tree.lods.at(-1).source.changed_area)+
    ' source-area change. The repair on this tree comes from the tighter pixel schedule; it is an asset-specific result.');
  lines.push('','Painted shelves show a concrete limit under the tighter contract. The four strict variants finish at '+
    strict.shelves.map(item=>item.final_triangles).join('/')+
    ' triangles (3B/3E/4B/4E), whereas frozen 8 px/variable-transition B finishes at 26 under a different pixel contract. '+
    'For 3 px B, LOD6 is '+strict.shelves[0].lod6_triangles+' triangles with source error '+
    fixed(strict.shelves[0].lod6_source_px)+'/'+fixed(strict.shelves[0].lod6_source_limit_px)+
    ' px and adjacent error '+fixed(strict.shelves[0].lod6_adjacent_px)+'/'+
    fixed(strict.shelves[0].lod6_adjacent_limit_px)+' px; LOD7 retains '+
    strict.shelves[0].lod7_triangles+' triangles.','',
    '| Shelves variant | Final tris | Source-audit rejects | Adjacent-audit rejects | Area-only audit rejects source/adjacent |',
    '|---|---:|---:|---:|---:|');
  for(const item of strict.shelves)lines.push('| '+item.source_cap_px+' px '+item.mode+' | '+item.final_triangles+
    ' | '+item.source_audit_rejections+' | '+item.adjacent_audit_rejections+' | '+
    item.area_only_source_audit_rejections+'/'+item.area_only_adjacent_audit_rejections+' |');
  lines.push('','Selected B proposal rejections when the source cap changes from 3 to 4 px:','',
    '| Asset | Source-search rejects 3→4 | Adjacent-search rejects 3→4 | Area-only source-audit rejects 3/4 |',
    '|---|---:|---:|---:|');
  for(const id of ['ph_dead_quiver_trunk','ph_grass_bermuda_01',
    'si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7']) {
    const lower=strict.pairs[0].control.rows.find(row=>row.id===id);
    const higher=strict.pairs[1].control.rows.find(row=>row.id===id);
    lines.push('| '+id+' | '+lower.search_rejections.source+'→'+higher.search_rejections.source+
      ' | '+lower.search_rejections.adjacent+'→'+higher.search_rejections.adjacent+
      ' | '+lower.area_only_audit_rejections.source+'/'+higher.area_only_audit_rejections.source+' |');
  }
  lines.push('','Across the different 3 px and 4 px source schedules, '+
    strict.cross_cap_identical_output_and_attributes.B+'/8 B and '+
    strict.cross_cap_identical_output_and_attributes.E+'/8 E output and attribute hashes are identical; '+
    strict.cross_cap_byte_identical_exports.B+'/8 B and '+strict.cross_cap_byte_identical_exports.E+
    '/8 E `chain.bin`/`chain.gltf` exports are byte-identical. This is a measured property of these eight assets, not a claim that the contracts are interchangeable. '+
    'The shelves counts show adjacent pixel pressure on that object. Tree and grass each have area-only source-audit rejections in both contracts, so the 50% area gate also constrains proposals even though their selected meshes remain below 50%. Rejection counts do not prove a global triangle optimum. The strict runs are separate pixel contracts from the frozen A/B/C and v2 E comparisons.');
}
lines.push('','## Where triangle headroom remains','',
  'SCORE averages triangle retention across scheduled LOD1–7 for each asset, then balances categories; final-LOD triangles alone do not determine it. On painted shelves, cap-only B ends at 26 triangles with 1,393 triangles across LOD1–7. Adaptive C ends at 34 triangles but totals 1,110, so C improves that asset’s chain ratio despite a denser final mesh. The [C trace](trace-extracts/shelves-adaptive/rows/ph_painted_wooden_shelves.json) shows a progressive proposal from a 46-triangle parent requesting 23 and achieving 24, rejected at adjacent search; C retains 34. The [B trace](trace-extracts/shelves-fixed/rows/ph_painted_wooden_shelves.json) shows a progressive 52→26 proposal accepted.','',
  'For Apollo hatch, the [QEM trace](trace-extracts/hatch-quadric/rows/si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7.json) requests 85 and 428 triangles directly from the 150,000-triangle source, but both stop at 856 after 26,673 link-condition rejections; the selected progressive final is 844. Post-hoc topology-relaxed D reaches 48 and passes its rotated dense hatch tail check, showing that topology constraints leave substantial object-specific headroom. D’s global pilot SCORE regression and rock/tree dense misses prevent a global switch. [Trace provenance and reproduction](trace-extracts/README.md).');
if(postHocFallbackPilot)lines.push('',
  'The conditional E fallback preserves B’s quadric path unless a large link-condition shortfall triggers one extra topology-relaxed proposal. It reaches 96 hatch triangles and 128 Bell X-1 triangles with five extra proposals across the corrected development pilot. '+
  (fallbackHatchDense&&fallbackBellDense?'Both selected exports pass their corrected rotated dense tail checks. ':'Their direct corrected dense tail checks are pending. ')+
  'The original 8 px/variable-transition tree export remains identical to B and retains its independent dense failure. E therefore demonstrates useful bounded, object-specific headroom without a validated global quality preset.');
if(compatibility)lines.push('','## Pre-change compatibility replay','',
  'The archived pre-change uncapped executable and fresh v3 uncapped executable produce identical positions/indices, all output attributes and canonical input attributes on '+compatibility.assets+'/'+compatibility.assets+' pilot assets: '+(compatibility.identical?'yes':'NO; see mismatches in analysis.json')+'. Old binary SHA-256: '+compatibility.binary_sha256+'.');
if(finalReplay)lines.push('',
  'After a JSON-manifest field was added, the final rebuilt binary replayed the capped tree with identical positions/indices, output attributes and canonical input attributes: '+
  (finalReplay.identical?'yes':'NO; see mismatches in analysis.json')+'. Its reported area limit is '+finalReplay.max_changed_area+
  '; binary SHA-256: '+finalReplay.binary_sha256+'. Frozen A/B validation uses this final rebuilt binary.');
if(dense) {
  lines.push('','## Independent rotated dense tree audit','',
    '[Rotated raw audit](dense-tree-cap-0.5.json) · [Default-seed raw audit](dense-tree-default-seed-cap-0.5.json) · auditor binary SHA-256: '+dense.binary_sha256+'.','',
    'The last three capped tree LODs were checked against source and predecessor with '+dense.audit.orthographic+'+'+dense.audit.perspective+
    ' cameras, seed '+dense.audit.seed+', '+dense.audit.supersample+'× sampling and refinement to '+dense.audit.max_supersample+'×. This is a separate camera/sampling contract and does not rewrite pilot SCORE. Overall: '+(dense.passed?'PASS':'FAIL')+'.','',
    '| LOD | Tris | Source area / cap | Adjacent area / cap | Source px / limit | Adjacent px / limit | Result |',
    '|---:|---:|---:|---:|---:|---:|---|');
  for(const lod of dense.lods)lines.push('| '+lod.level+' | '+lod.triangles+' | '+pct(lod.source.changed_area)+' / '+pct(lod.max_changed_area)+
    ' | '+pct(lod.adjacent.changed_area)+' / '+pct(lod.max_changed_area)+' | '+fixed(lod.source.error_px)+' / '+fixed(lod.source_limit)+
    ' | '+fixed(lod.adjacent.error_px)+' / '+fixed(lod.transition_limit)+' | '+(lod.source.passed&&lod.adjacent.passed?'pass':'FAIL')+' |');
  if(denseDefault)lines.push('',
    'A second dense check with the original audit rotation seed '+denseDefault.audit.seed+' and identical camera count and sampling passes all six gates; its LOD7 source area maximum is '+
    pct(denseDefault.lods.find(lod=>lod.level===7)?.source.changed_area??null)+'. The rotated check changes only the seed and exposes a failing view.');
  if(dense.failures.length)lines.push('',
    'The first failure is LOD '+dense.failures[0].level+' '+dense.failures[0].kind+' area '+pct(dense.failures[0].changed_area)+
    ' at view '+dense.failures[0].changed_area_worst_view+' after '+dense.failures[0].views_evaluated+' views; its pixel-distance gate passes. The 0.5 preset is not ready for promotion under this independent audit. Finite-camera acceptance does not establish all-view robustness.');
}
if(exploratory)lines.push('','## Exploratory cap check (unscored)','',
  'A one-tree development follow-up tested stricter caps without changing the frozen A/B/C comparison. At 0.45, the reducer selected the same '+
  exploratory.cap_045.final_triangles+'-triangle output hash as the 0.5 cap, so its rotated dense miss persists. At 0.4, it selected a different '+
  exploratory.cap_04.final_triangles+'-triangle mesh with '+pct(exploratory.cap_04.configured_source_area)+' configured final source area, but its '+
  '[rotated dense audit](dense-tree-dev-cap-0.4-rotated.json) failed at '+pct(exploratory.cap_04.rotated_lod7_source_area)+
  ' against the 40% cap (LOD7 source view '+exploratory.cap_04.rotated_lod7_worst_view+'). These single-asset probes have no SCORE and do not select a new preset.');
if(validation) {
  lines.push('','## Validation','',
    '| Variant | Complete | SCORE | Final retained, category mean | Worst final source area | Worst source area at any LOD | Fallbacks | Failed | All delivered audits pass |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|');
  for(const run of [validation.uncapped,validation.cap,...(validation.adaptive?[validation.adaptive]:[])])lines.push('| '+run.name+' | '+run.summary.completed+'/'+run.summary.expected+' | '+fixed(run.summary.score)+
    ' | '+pct(run.category_balanced_final_ratio)+' | '+pct(Math.max(0,...run.rows.map(row=>row.final_source_area??0)))+' | '+pct(Math.max(0,...run.rows.map(row=>row.source_area?.value??0)))+
    ' | '+run.summary.fallbacks+' | '+run.failures.length+' | '+(run.all_delivered_audits_pass?'yes':'no')+' |');
  lines.push('','Cap validation quality cost: '+fixed(validationScoreCost)+' SCORE points; qualification: '+(capValidation?'pass':'fail or incomplete')+'.');
  const overLimit=run=>run.rows.filter(row=>(row.final_source_area??0)>0.5).length;
  lines.push('',
    'Final LODs above 50% source-area change: '+overLimit(validation.uncapped)+'/'+validation.uncapped.summary.expected+
    ' uncapped versus '+overLimit(validation.cap)+'/'+validation.cap.summary.expected+' capped. This verifies improvement on the frozen validation split under the configured cameras.');
  lines.push('',
    'The uncapped run has '+validationChange.uncapped_area_breaches.length+' over-cap source/adjacent LOD pairs across '+
    new Set(validationChange.uncapped_area_breaches.map(item=>item.id)).size+' assets; capped B repairs them. Its worst adjacent area is '+
    pct(validationChange.capped_worst_adjacent_area)+'. Area alone rejected '+validationChange.capped_area_only_audit_rejections.source+
    ' source-audit and '+validationChange.capped_area_only_audit_rejections.adjacent+' adjacent-audit proposals. '+
    validationChange.identical_output_and_attributes+'/'+validation.cap.summary.expected+' output and attribute hashes remain identical, '+
    validationChange.identical_canonical_attributes+'/'+validation.cap.summary.expected+' canonical input attribute hashes match, and '+
    (validationChange.no_final_source_area_worsened?'no':'some')+' final source-area result worsened.');
  if(validationChange.changed_rows.length) {
    lines.push('','Changed validation exports:','',
      '| Asset | Final tris A→B | Final source area A→B |',
      '|---|---:|---:|');
    for(const row of validationChange.changed_rows)lines.push('| '+row.id+' | '+row.uncapped_final_triangles+'→'+row.capped_final_triangles+
      ' | '+pct(row.uncapped_final_source_area)+'→'+pct(row.capped_final_source_area)+' |');
  }
  if(validation.adaptive)lines.push('','Adaptive validation qualification: '+(adaptiveValidation?'pass':'fail or incomplete')+'.');
}
lines.push('','## Decision','',
  'The 0.5 cap passes the configured pilot and '+(validation?'20-asset validation':'pilot')+' screens with a small SCORE cost, but the rotated dense tree check fails its area limit. '+
  'Keep the 0.5 gate available as an experimental opt-in; do not label it an independently validated quality preset. Adaptive targets miss the pilot gain threshold. '+
  (postHoc?'Topology relaxation improves selected objects but regresses aggregate chain SCORE and fails dense checks on other objects. ':'')+
  (postHocFallbackPilot?'Conditional topology fallback E reduces the hatch and Bell triangle counts, but the original 8 px/variable-transition E tree fails the rotated dense gate. Keep E opt-in. ':'')+
  (strict?'The separate strict 2 px progressive/3–4 px source scenarios pass their configured pilot audits and the sampled rotated tail checks, but have no held-out validation or all-view bound; retain them as research scenarios.':'') );
lines.push('','## Provenance','',
  '| Run | Binary SHA-256 | Source tree SHA-256 | Config SHA-256 | Protocol SHA-256 |',
  '|---|---|---|---|---|');
for(const run of [uncapped,cap,adaptive,...(postHoc?[postHoc.run]:[]),
  ...(postHocFallbackPilot?[...(postHocFallbackPilot.strict_same_binary_comparison?[postHocFallbackPilot.control]:[]),postHocFallbackPilot.run]:[]),
  ...(validation?[validation.uncapped,validation.cap,...(validation.adaptive?[validation.adaptive]:[])]:[]),
  ...(postHocFallbackValidation?[postHocFallbackValidation.control,postHocFallbackValidation.experimental]:[]),
  ...(strict?strict.pairs.flatMap(pair=>[pair.control,pair.experimental]):[])])
  lines.push('| '+run.name+' | '+run.binary_sha256+' | '+(run.build_stamp?.source_tree_sha256??'—')+' | '+run.config_sha256+' | '+run.protocol_sha256+' |');
lines.push('','Run hashes, limits, worst views, and all measured asset summaries are in analysis.json; raw benchmark rows retain each LOD and rejection. Resident bytes include source vertices, added vertices and indices. Generation time includes search and audits but excludes import/export. Process RSS is a cumulative high-water mark, so its per-asset row is not an isolated memory cost. Timings include shared-workstation noise.','');
await mkdir(args[3],{recursive:true});
await writeFile(join(args[3],'analysis.json'),JSON.stringify(report,null,2)+'\n');
await writeFile(join(args[3],'REPORT.md'),lines.join('\n'));
console.log(JSON.stringify(report.decisions,null,2));
