// Bounded, resumable first experiment. Run under the Nix neural shell/systemd user service.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn, execFileSync} from 'node:child_process';
import {auditProgress} from './audit.mjs';
const root=process.cwd(), run=path.resolve(process.argv[2]??'runs/neural/first-pass');
fs.mkdirSync(run,{recursive:true});
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const write=(p,j)=>{fs.writeFileSync(p+'.part',JSON.stringify(j,null,2)+'\n');fs.renameSync(p+'.part',p);};
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const statusPath=path.join(run,'status.json');
let status=fs.existsSync(statusPath)?read(statusPath):{started_at:new Date().toISOString(),deadline_ms:Date.now()+10*3600000,phase:'starting'};
let child, stopping=false;
const stamp=(phase,extra={})=>{status={...status,...extra,phase,updated_at:new Date().toISOString(),pid:process.pid};write(statusPath,status);console.log(JSON.stringify(status));};
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{stopping=true;child?.kill('SIGTERM');stamp('stopping',{signal});});
const remaining=()=>Math.max(0,status.deadline_ms-Date.now());
const execute=async(executable,args,label)=>{
  if(stopping||remaining()<60000)throw new Error('experiment deadline or cancellation reached');
  stamp(label,{command:[executable,...args]});const output=fs.openSync(path.join(run,label.replaceAll('/','-')+'.log'),'a');
  const started=Date.now();child=spawn(executable,args,{cwd:root,env:process.env,stdio:['ignore',output,output]});
  stamp(label,{child_pid:child.pid});
  const deadline=setTimeout(()=>{stopping=true;child?.kill('SIGTERM');},remaining());
  const code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',resolve);});
  clearTimeout(deadline);fs.closeSync(output);child=undefined;
  status.timings??=[];status.timings.push({label,seconds:(Date.now()-started)/1000,code});
  if(code!==0&&code!==2&&code!==3)throw new Error(`${label} failed with exit ${code}; see ${label}.log`);
  return code;
};
const binaries=path.join(run,'bin');fs.mkdirSync(binaries,{recursive:true});
for(const name of ['blitz','blitz-neural-prepare','blitz-neural-train']){
  const target=path.join(binaries,name);if(!fs.existsSync(target)){fs.copyFileSync(path.join(root,'build/neural',name),target);fs.chmodSync(target,0o755);}
}
const provenancePath=path.join(run,'provenance.json');
if(!fs.existsSync(provenancePath)){
  const files=execFileSync('rg',['--files','src','include','tools','tests','cmake','CMakeLists.txt','flake.nix','flake.lock','research/neural'],{encoding:'utf8'}).trim().split('\n').sort();
  write(provenancePath,{git_revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),source_files:files.map(file=>({file,sha256:hash(file)})),
    binaries:Object.fromEntries(['blitz','blitz-neural-prepare','blitz-neural-train'].map(n=>[n,hash(path.join(binaries,n))])),
    gpu:execFileSync('nvidia-smi',['--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],{encoding:'utf8'}).trim(),
    protocol_sha256:hash('research/PROTOCOL.md'),config_sha256:hash('research/neural/first-pass.json'),corpus_sha256:hash('research/corpus.json')});
}
const minutes=()=>String(Math.min(50,Math.max(1,Math.floor((remaining()-30000)/60000))));
async function prepare(directory,model){
  let previous=-1;
  for(;;){const indexPath=path.join(directory,'index.json');if(fs.existsSync(indexPath)&&read(indexPath).complete)return;
    const args=['research/corpus.json','research/pilot.json',directory,'--minutes',minutes()];if(model)args.push('--model',model);
    await execute(path.join(binaries,'blitz-neural-prepare'),args,'prepare-'+path.basename(directory));
    const index=read(indexPath);if(index.complete)return;if(index.assets.length<=previous)throw new Error('preparation made no progress');previous=index.assets.length;
  }
}
async function train(dataset,directory,steps,initialize){
  for(;;){const latest=path.join(directory,'latest.json');if(fs.existsSync(latest)&&read(latest).step>=steps)return read(latest);
    const args=[dataset,directory,'--steps',String(steps),'--segment-minutes',minutes(),'--checkpoint-every','100','--core','4096','--batch','4'];if(initialize)args.push('--initialize',initialize);
    await execute(path.join(binaries,'blitz-neural-train'),args,'train-'+path.basename(directory));
    if(!fs.existsSync(latest))throw new Error('training ended without a checkpoint');
  }
}
async function benchmark(manifest,split,model,directory,label){
  let previous=-1;
  for(;;){const summary=path.join(directory,'summary.json');if(fs.existsSync(summary)){const saved=read(summary);auditProgress(saved);if(saved.complete)return saved;}
    await execute(path.join(binaries,'blitz'),['bench',manifest,'research/neural/first-pass.json',directory,'--split',split,'--neural-model',model,'--minutes',minutes()],label);
    previous=auditProgress(read(summary),previous);
  }
}
try {
  const data=path.join(run,'data'),training=path.join(run,'bootstrap');
  await prepare(data);
  const health=await train(data,training,100);
  const metrics=fs.readFileSync(path.join(training,'metrics.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  const mean=a=>a.reduce((n,m)=>n+m.loss,0)/a.length;
  if(!health.finite||!health.restored||health.native_max_abs_error>2e-4||health.parameter_change<=0||mean(metrics.slice(-25))>=mean(metrics.slice(0,25)))throw new Error('training health gate failed');
  const readinessModel=path.join(training,health.model), readinessLabel='pilot-readiness-'+hash(readinessModel);
  await benchmark('research/pilot.json','development',readinessModel,path.join(run,readinessLabel),readinessLabel);
  write(path.join(run,'health.json'),{...health,mean_first_25:mean(metrics.slice(0,25)),mean_last_25:mean(metrics.slice(-25)),full_run_steps:5120,full_run_eta_seconds:health.seconds_per_step*5120});
  stamp('healthy',{health:'health.json',checkpoint:path.join('bootstrap',health.checkpoint),training_eta_seconds:health.seconds_per_step*(5120-health.step)});
  const initial=await train(data,training,5120);const initialModel=path.join(training,initial.model);
  const refinedData=path.join(run,'refined-data'),refinedTrain=path.join(run,'refined');
  await prepare(refinedData,initialModel);const refined=await train(refinedData,refinedTrain,5120,initialModel);const refinedModel=path.join(refinedTrain,refined.model);
  const pilot=await benchmark('research/pilot.json','development',initialModel,path.join(run,'pilot-bootstrap'),'pilot-bootstrap');
  const pilotRefined=await benchmark('research/pilot.json','development',refinedModel,path.join(run,'pilot-refined'),'pilot-refined');
  const candidates=[{model:initialModel,pilot},{model:refinedModel,pilot:pilotRefined}];
  candidates.sort((a,b)=>b.pilot.score-a.pilot.score);const promising=candidates[0];
  if(promising.pilot.fallbacks>=promising.pilot.expected)throw new Error('pilot contains only unreduced fallbacks; retain measurements for the next research round');
  // Validation chooses between checkpoints. Held-out assets are never opened here.
  for(const [i,candidate] of candidates.entries())candidate.validation=await benchmark('research/corpus.json','validation',candidate.model,path.join(run,'validation-'+i),'validation-'+i);
  candidates.sort((a,b)=>b.validation.score-a.validation.score);const best=candidates[0];
  fs.copyFileSync(best.model,path.join(run,'selected.blzn'));
  const report={complete:true,release_approved:false,selection:'Highest complete validation SCORE under the frozen first-pass configuration; no CPU performance threshold.',selected_model:'selected.blzn',model_sha256:hash(path.join(run,'selected.blzn')),training_assets:read(path.join(data,'index.json')).assets.length,candidates,health:read(path.join(run,'health.json')),timings:status.timings,held_out_used:false};
  write(path.join(run,'report.json'),report);stamp('complete',{report:'report.json',model:'selected.blzn'});
}catch(error){
  write(path.join(run,'report.json'),{complete:false,error:String(error),audit:error.audit??null,release_approved:false,timings:status.timings});
  stamp(stopping||remaining()<60000?'incomplete':error.audit?.blocked_assets?.length?'audit-resource-limited':'failed',{error:String(error),release_approved:false,report:'report.json'});process.exitCode=1;
}
