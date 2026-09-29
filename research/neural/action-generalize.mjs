// Bounded continuation after the independently recorded one-mesh proof.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {read,write} from './runpod-api.mjs';
import {endpointDecision,actionHealth} from './action-gates.mjs';
const evidence=fileURLToPath(new URL('./evidence/action-v2',import.meta.url));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
export async function generalize({root,execute,deadline,phase,reusePrepared=true}){
  const manifest=read(evidence+'/manifest.json'),seeds=[101,202,303];
  for(const seed of seeds){
    if(!endpointDecision(read(evidence+`/history-${seed}.json`)).passed)throw new Error('Missing persisted endpoint proof');
    const file=manifest.files.find(f=>f.seed===seed&&f.step===8192);
    if(!file||file.path!==`seed-${seed}-step-8192.blzn`||hash(evidence+'/'+file.path)!==file.sha256)throw new Error('Endpoint evidence model changed');
  }
  const result={complete:false,endpoint_gate_passed:true,quality_proven:false,generalization_gate_passed:false,stages:[]};
  const data=root+'/curriculum',datasets=['endpoint-proof'];
  fs.mkdirSync(data,{recursive:true});
  const conditions=[{pixels:16,states:16,previous:4,adjacent:2},{pixels:64,states:16,previous:4,adjacent:3},{pixels:256,states:8,previous:2,adjacent:2}];
  const saved=fileURLToPath(new URL('./evidence/action-curriculum-v2/data',import.meta.url));
  if(reusePrepared&&fs.existsSync(saved+'/index.json')){
    const expected=['endpoint-proof',...['ph_sweet_potato','ph_painted_wooden_bench'].flatMap(a=>conditions.map(c=>a+'-'+c.pixels))];
    if(JSON.stringify(read(saved+'/index.json').datasets)!==JSON.stringify(expected))throw new Error('Saved curriculum selection changed');
    for(const name of expected){const d=saved+'/'+name,j=read(d+'/index.json');
      if(!j.complete||!j.reference_confirmed||hash(d+'/actions.bin')!==j.sha256||hash(d+'/contract.json')!==j.contract_sha256)throw new Error('Saved curriculum is incomplete or changed');
      if(name!=='endpoint-proof'&&(!j.preceding_lod_emitted||j.previous_triangles>=j.source_triangles))throw new Error('Saved preceding LOD is missing');
    }
    fs.cpSync(saved,data,{recursive:true});datasets.splice(0,datasets.length,...expected);
  }else{
  fs.cpSync(evidence+'/proof-data',data+'/endpoint-proof',{recursive:true});
  write(data+'/curriculum.json',{assets:['ph_sweet_potato','ph_painted_wooden_bench'],conditions,source_limit:3,policy:manifest.files.find(f=>f.seed===101&&f.step===8192).sha256});
  phase('preparation');
  for(const asset of ['ph_sweet_potato','ph_painted_wooden_bench'])for(const c of conditions){
    const name=asset+'-'+c.pixels,out=data+'/'+name;
    await execute('blitz-neural-action-prepare',[asset,out,'--states',String(c.states),'--previous-steps',String(c.previous),'--pixels',String(c.pixels),'--source-limit','3','--adjacent-limit',String(c.adjacent),'--model',evidence+'/seed-101-step-8192.blzn','--gpu-memory-mib','16384','--minutes','4'],root+'/prepare.log',4.5);
    const index=read(out+'/index.json');
    if(!index.complete||!index.reference_confirmed||!index.preceding_lod_emitted||index.previous_triangles>=index.source_triangles)throw new Error('Curriculum lacks an audited preceding LOD');
    datasets.push(name);write(data+'/progress.json',{complete:false,datasets,at:Date.now()});
  }
  }
  write(data+'/index.json',{datasets});
  write(data+'/progress.json',{complete:true,datasets,at:Date.now()});
  let previous,previousDuration=0;
  // Two checkpoints are sufficient to test persistence. A failed pilot remains
  // visible, and no third tuning stage or more expensive model is automatic.
  for(let stage=1;stage<=2;++stage){
    if(Date.now()+Math.max(60000,previousDuration+60000)>=deadline){result.stop_reason='insufficient_time_for_matched_pilot';break;}
    const models=[],health=[];
    for(const seed of seeds){
      phase('training');const run=root+'/seed-'+seed;
      await execute('blitz-neural-action-train',[data,run,'--steps',String(stage*8192),'--minutes','4','--batch','512','--seed',String(seed)],root+`/train-${seed}.log`,4.5);
      const h=read(run+'/latest.json');if(!actionHealth(h))throw new Error('Generalization training failed numerical or restore health');
      health.push({seed,...h});models.push(run+'/'+h.model);
      write(root+'/progress.json',{phase:'training',stage,health,at:Date.now()});
    }
    phase('audit');write(root+'/progress.json',{phase:'audit',stage,at:Date.now()});const began=Date.now(),out=root+'/pilot-'+stage;
    const minutes=Math.min(45,(deadline-Date.now()-20000)/60000);
    if(minutes<1){result.stop_reason='insufficient_time_for_matched_pilot';break;}
    const args=['research/neural/action-pilot.mjs',out,...models,'--minutes',String(minutes),'--action-trials','8','--action-batch','32','--gpu-memory-mib','16384'];
    if(previous)args.push('--previous',previous);
    await execute('node',args,root+`/pilot-${stage}.log`,minutes+.1);
    const report=read(out+'/report.json');previousDuration=Date.now()-began;
    result.stages.push({stage,health,report:out+'/report.json',gate:report.gate,complete:report.complete,persisted:report.persisted});
    write(root+'/progress.json',{phase:'audit',...result,at:Date.now()});
    if(!report.complete){result.stop_reason='incomplete_matched_pilot';break;}
    previous=out+'/report.json';
    if(report.persisted){result.generalization_gate_passed=true;result.stop_reason='matched_advantage_persisted';break;}
    if(stage===2)result.stop_reason='two_checkpoints_without_persisted_advantage';
  }
  result.complete=result.generalization_gate_passed||result.stages.length===2&&result.stages.every(s=>s.complete);
  result.next_phase=result.generalization_gate_passed?'profile_training_and_audits':'inspect_matched_pilot_evidence';
  return result;
}
