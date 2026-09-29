// Build an offline explanation from the recorded experiment, without running it.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const root=fileURLToPath(new URL('..',import.meta.url));
const output=path.resolve(process.argv[2]??path.join(root,'research/neural/viewer'));
const evidence='research/neural/evidence/action-screening-v3';
const files=new Map();
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function read(p){const bytes=fs.readFileSync(path.join(root,p));files.set(p,{path:p,sha256:sha(bytes)});return JSON.parse(bytes);}
const source=p=>'../evidence/action-screening-v3/'+p;
const sum=a=>a.reduce((total,n)=>{assert.ok(Number.isFinite(n)&&n>=0,'Invalid recorded duration');return total+n;},0);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-6*Math.max(1,Math.abs(b)),`Inconsistent totals: ${a} vs ${b}`);
const result=read(evidence+'/final/result.json'),outcome=read(evidence+'/final/outcome.json');
const baseline=read(evidence+'/action-v2-staged/baseline-health.json');
const curriculum=read(evidence+'/action-v2-staged/curriculum/index.json');
const reports=[1,2].map(i=>read(evidence+`/final/pilot-${i}/report.json`));
const contract=read(evidence+'/final/seed-101/contract.json');
assert.ok(result.complete&&outcome.complete&&baseline.healthy&&baseline.summary.complete);
assert.equal(result.stages.length,2);assert.ok(reports.every(r=>r.complete&&r.runs.every(m=>m.health_complete&&m.summary.complete)));
const health=result.stages.flatMap(s=>s.health);
assert.equal(health.length,6);assert.ok(health.every(h=>h.complete&&h.finite&&h.restored&&h.optimizer_restored));
assert.ok(health.every(h=>h.states===health[0].states));
near(sum(health.map(h=>h.steps_this_segment)),result.training_steps);
assert.equal(result.training_steps,outcome.remote_optimizer_steps);
const fresh=curriculum.datasets.filter(n=>n.startsWith('fresh-')).map(n=>{
  const p='action-v2-staged/curriculum/'+n+'/index.json',j=read(evidence+'/'+p);
  assert.ok(j.complete&&j.reference_confirmed);return {name:n,seconds:j.seconds,states:j.states,source:source(p)};
});
assert.ok(fresh.length>0);
const training=sum(health.map(h=>h.seconds));
const comparisons=reports.map(r=>sum(r.runs.map(m=>m.wall_seconds)));
const audit=sum([baseline.summary.seconds,...comparisons]),labels=sum(fresh.map(f=>f.seconds));
const total=outcome.rental_minutes*60,other=total-audit-training-labels;
assert.ok(other>=0);near(sum([audit,training,labels,other]),total);
const rock=reports[1].runs.find(r=>r.name==='learned-0').rows.find(r=>r.id==='ph_moon_rock_02');
const controlRock=reports[1].runs.find(r=>r.name==='constant').rows.find(r=>r.id===rock.id);
assert.ok(rock.complete&&controlRock.complete);
const n=rock.neural,rockAudit=sum([n.gpu_audit_seconds,n.gpu_confirmation_seconds]),rockOther=rock.seconds-rockAudit-n.inference_seconds;
assert.ok(rockOther>=0);
const lods=read(evidence+'/final/pilot-2/learned-0/meshes/'+rock.id+'/lods.json');
const sampled=lods.lods[1],views=sampled.source.views_evaluated,ss=sampled.source.supersample;
assert.ok(sampled.source.complete&&sampled.source.passed&&views>0&&ss>0);
const side=Math.ceil(sampled.screen_pixels+8)*ss;
// This frozen run used architecture 2: 80 -> 64 -> 64 -> 3, including biases.
const architecture=[80,64,64,3],parameters=sum(architecture.slice(1).map((out,i)=>(architecture[i]+1)*out));
const data={
  revision:outcome.revision,date:outcome.finished_utc.slice(0,10),states:health[0].states,
  seeds:[...new Set(health.map(h=>h.seed))],updates:result.training_steps,checkpoints:result.stages.map(s=>s.health[0].step),
  batch:contract.batch,precision:contract.precision,architecture,parameters,
  busy:{training:outcome.phases.training.mean_gpu_busy_percent,audit:outcome.phases.audit.mean_gpu_busy_percent},
  auditExample:{pixels:sampled.screen_pixels,views,supersample:ss,side,samples:side*side},
  learning:reports.map((r,i)=>({step:result.stages[i].health[0].step,assets:r.runs.find(m=>m.name==='learned-0').rows.map(a=>({id:a.id,mean_reduction:1-a.ratio}))})),
  charts:{
    rental:{title:'Entire rental',seconds:total,unit:'minutes',source:source('final/outcome.json'),segments:[
      {id:'evaluation',name:'Baseline + mesh comparisons',seconds:audit,color:'#bf642d',kind:'Recorded stages',
        text:'One baseline plus two checkpoint comparisons. These timings include LOD generation, its visual audits, and small surrounding process costs. They are not pure GPU kernel time.',
        parts:[{name:'Baseline',seconds:baseline.summary.seconds},{name:'First comparison',seconds:comparisons[0]},{name:'Second comparison',seconds:comparisons[1]}],
        sources:[source('action-v2-staged/baseline-health.json'),source('final/pilot-1/report.json'),source('final/pilot-2/report.json')]},
      {id:'training',name:'Training + checkpoint checks',seconds:training,color:'#287f67',kind:'Recorded stages',
        text:'Six trainer segments across three seeds. Includes optimizer updates and periodic checkpoint/export/restore checks. Input tensors and labels stay on the GPU; no mesh audit runs inside each optimizer update.',sources:[source('final/result.json')]},
      {id:'labels',name:'Fresh label preparation',seconds:labels,color:'#7660ad',kind:'Recorded stages',
        text:'Four new training shards. Earlier verified shards were reused. Labels describe queried trial edits; unknown choices are not treated as failures. These durations include the preparer’s reference confirmations.',parts:fresh.map(f=>({name:f.name.replace('fresh-ph_',''),seconds:f.seconds})),sources:fresh.map(f=>f.source)},
      {id:'other',name:'Setup / processes / collection remainder',seconds:other,color:'#8193a0',kind:'Derived remainder',
        text:'Rental duration minus the recorded stages above. This includes setup, builds/tests, process startup, orchestration and collection, but their individual shares were not measured here. It must not be read as idle GPU time or a measured build duration.',sources:[source('final/outcome.json'),source('final/result.json')]}
    ]},
    rock:{title:'Final moon-rock generation',seconds:rock.seconds,unit:'seconds',source:source('final/pilot-2/report.json'),segments:[
      {id:'audit',name:'GPU audits + confirmation',seconds:rockAudit,color:'#bf642d',kind:'Recorded stages',
        text:`${n.gpu_evaluations} GPU evaluations were executed, versus ${controlRock.neural.gpu_evaluations} for the constant control. An evaluation can fail before visiting all cameras. Passing comparisons and changed geometry can require more work. Stage time includes host synchronization.`,
        parts:[{name:'Proposal / chain audits',seconds:n.gpu_audit_seconds},{name:'Final confirmation',seconds:n.gpu_confirmation_seconds}],sources:[source('final/pilot-2/report.json')]},
      {id:'inference',name:'Network ranking / inference',seconds:n.inference_seconds,color:'#287f67',kind:'Recorded stage',
        text:'Scoring legal candidate actions is a tiny part of generation time. Making this stage faster alone would scarcely reduce this mesh’s total bake time.',sources:[source('final/pilot-2/report.json')]},
      {id:'remaining',name:'Topology + other generation work',seconds:rockOther,color:'#8193a0',kind:'Derived remainder',
        text:'Total asset time minus audit/confirmation and inference. Includes topology edits, feature preparation, loading/export and other overhead. It is not a direct measure of all CPU activity: audit stage time also includes host work and synchronization.',sources:[source('final/pilot-2/report.json')]}
    ]}
  },
  references:{evidence:source('README.md'),training:source('final/result.json'),comparison:source('final/pilot-2/report.json'),
    implementation:'../V2.md',gpu:'../AUDIT.md',acceleration:'../ACCELERATION.md',spec:'../../../docs/SPEC.md'},
  methodology:{rental:'Audit = baseline summary seconds + both reports’ method wall_seconds. Training = sum of six health.seconds. Labels = sum of four fresh shard index.seconds. Remainder = rental_minutes × 60 minus those three totals.',
    rock:'Audit = gpu_audit_seconds + gpu_confirmation_seconds. Inference = inference_seconds. Remainder = asset seconds minus audit and inference.',
    sample:'Illustrative raster sample count = (ceil(screen_pixels + 8) × supersample)² for a saved audited LOD. This is one image at one sampling level, not a count of samples actually processed across the run.'}
};
for(const c of Object.values(data.charts))near(sum(c.segments.map(s=>s.seconds)),c.seconds);
const template=fs.readFileSync(path.join(root,'tools/neural-loop.html'),'utf8');assert.equal(template.split('@LOOP_DATA@').length,2);
const manifest={revision:data.revision,files:[...files.values()],methodology:data.methodology};
fs.mkdirSync(output,{recursive:true});
fs.writeFileSync(path.join(output,'learning-loop.html'),template.replace('@LOOP_DATA@',JSON.stringify(data).replaceAll('<','\\u003c')));
fs.writeFileSync(path.join(output,'learning-loop-data.json'),JSON.stringify(data,null,2)+'\n');
fs.writeFileSync(path.join(output,'learning-loop-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({page:path.join(output,'learning-loop.html'),inputs:files.size,rental_seconds:total,training_seconds:training,evaluation_seconds:audit,labels_seconds:labels,remainder_seconds:other,rock_audit_seconds:rockAudit},null,2));
