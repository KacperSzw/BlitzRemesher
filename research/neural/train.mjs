// Continue a prepared curriculum with measured CUDA saturation and periodic quality gates.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn, execFileSync} from 'node:child_process';

if(process.argv.length<5)throw new Error('train.mjs RUN DATASET INITIAL_MODEL [DEADLINE_MS]');
const root=process.cwd(), run=path.resolve(process.argv[2]);
const dataset=path.resolve(process.argv[3]), initialize=path.resolve(process.argv[4]);
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const write=(p,j)=>{fs.writeFileSync(p+'.part',JSON.stringify(j,null,2)+'\n');fs.renameSync(p+'.part',p);};
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
fs.mkdirSync(run,{recursive:true});
const configPath=path.join(run,'training.json'), auditPath=path.join(run,'audit.json');
if(!fs.existsSync(configPath))fs.copyFileSync('research/neural/sustained.json',configPath);
if(!fs.existsSync(auditPath))fs.copyFileSync('research/neural/first-pass.json',auditPath);
const config=read(configPath), statusPath=path.join(run,'status.json');
let status=fs.existsSync(statusPath)?read(statusPath):{
  started_at:new Date().toISOString(),deadline_ms:Number(process.argv[5]??Date.now()+config.hours*3600000),timings:[]
};
const stamp=(phase,extra={})=>{
  status={...status,...extra,phase,pid:process.pid,updated_at:new Date().toISOString()};
  write(statusPath,status);console.log(JSON.stringify(status));
};
const remaining=()=>Math.max(0,status.deadline_ms-Date.now());
const minutes=()=>String(Math.min(50,Math.max(1,Math.floor((remaining()-30000)/60000))));
const bin=path.join(run,'bin'), training=path.join(run,'training');
fs.mkdirSync(bin,{recursive:true});
for(const name of ['blitz','blitz-neural-train'])if(!fs.existsSync(path.join(bin,name))){
  fs.copyFileSync(path.join(root,'build/neural',name),path.join(bin,name));fs.chmodSync(path.join(bin,name),0o755);
}
const contract={dataset,index_sha256:hash(path.join(dataset,'index.json')),initialize,initialize_sha256:hash(initialize),
  training_sha256:hash(configPath),audit_sha256:hash(auditPath),pilot_sha256:hash('research/pilot.json'),
  corpus_sha256:hash('research/corpus.json'),protocol_sha256:hash('research/PROTOCOL.md'),
  binaries:Object.fromEntries(['blitz','blitz-neural-train'].map(n=>[n,hash(path.join(bin,n))]))};
const provenancePath=path.join(run,'provenance.json');
if(fs.existsSync(provenancePath)){
  if(JSON.stringify(read(provenancePath).contract)!==JSON.stringify(contract))throw new Error('Run inputs changed; use a new directory');
}else{
  const files=execFileSync('rg',['--files','src','include','tools','tests','cmake','CMakeLists.txt','flake.nix','flake.lock','research/neural'],{encoding:'utf8'}).trim().split('\n').sort();
  const archive=path.join(run,'source.tar.gz');execFileSync('tar',['-czf',archive,'--null','-T','-'],{input:files.join('\0')+'\0'});
  write(provenancePath,{contract,git_revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),source_archive_sha256:hash(archive),
    source_files:files.map(file=>({file,sha256:hash(file)})),
    gpu:execFileSync('nvidia-smi',['--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],{encoding:'utf8'}).trim()});
}
let child,stopping=false;
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{stopping=true;child?.kill('SIGTERM');stamp('stopping',{signal});});
let pending='',samples=[];
const telemetry=fs.createWriteStream(path.join(run,'gpu.jsonl'),{flags:'a'});
const monitor=spawn('nvidia-smi',['--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits','-l','1']);
monitor.stdout.on('data',chunk=>{
  pending+=chunk.toString();let at;
  while((at=pending.indexOf('\n'))>=0){
    const values=pending.slice(0,at).split(',').map(Number);pending=pending.slice(at+1);
    if(values.length!==5||values.some(v=>!Number.isFinite(v)))continue;
    const [gpu,memory_busy,memory_mib,power_w,temperature_c]=values;
    const row={at:new Date().toISOString(),phase:status.phase,gpu,memory_busy,memory_mib,power_w,temperature_c};
    telemetry.write(JSON.stringify(row)+'\n');samples.push(row);if(samples.length>300)samples.shift();
  }
});
monitor.on('error',error=>{stopping=true;child?.kill('SIGTERM');stamp('failed',{error:String(error)});});
async function execute(executable,args,label){
  if(stopping||remaining()<60000)throw new Error('Experiment deadline or cancellation reached');
  const output=fs.openSync(path.join(run,label+'.log'),'a'), start=Date.now();
  child=spawn(executable,args,{cwd:root,env:process.env,stdio:['ignore',output,output]});
  stamp(label,{command:[executable,...args],child_pid:child.pid});
  const timer=setTimeout(()=>{stopping=true;child?.kill('SIGTERM');},remaining());
  let code;
  try{code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',resolve);});}
  finally{clearTimeout(timer);fs.closeSync(output);child=undefined;}
  status.timings.push({label,seconds:(Date.now()-start)/1000,code});
  if(stopping)throw new Error('Experiment interrupted; saved checkpoints and audit rows can resume');
  if(code!==0&&code!==2)throw new Error(`${label} failed with exit ${code}; see ${label}.log`);
}
async function train(steps){
  let previous=-1;
  for(;;){
    const latest=path.join(training,'latest.json');
    if(fs.existsSync(latest)&&read(latest).step>=steps)return read(latest);
    await execute(path.join(bin,'blitz-neural-train'),[dataset,training,'--initialize',initialize,'--steps',String(steps),
      '--segment-minutes',minutes(),'--checkpoint-every',String(config.checkpoint_every),'--core',String(config.core),'--batch',String(config.batch)],'training');
    const health=read(latest);if(health.step<=previous)throw new Error('Training made no progress');previous=health.step;
  }
}
async function audit(manifest,split,model,label){
  const directory=path.join(run,label), summary=path.join(directory,'summary.json');let previous=-1;
  for(;;){
    if(fs.existsSync(summary)&&read(summary).complete)return read(summary);
    await execute(path.join(bin,'blitz'),['bench',manifest,auditPath,directory,'--split',split,'--neural-model',model,'--minutes',minutes()],label);
    if(!fs.existsSync(summary))throw new Error('Audit ended without a summary');
    const progress=read(summary);if(!progress.complete&&progress.completed<=previous)throw new Error('Audit made no progress');previous=progress.completed;
  }
}
try{
  const healthPath=path.join(run,'health.json');
  if(!fs.existsSync(healthPath)){
    const health=await train(config.health_steps);
    const metrics=fs.readFileSync(path.join(training,'metrics.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
    const mean=values=>values.reduce((sum,value)=>sum+value,0)/values.length;
    const window=samples.filter(s=>s.phase==='training').slice(-60), sorted=window.map(s=>s.gpu).sort((a,b)=>a-b);
    const first=mean(metrics.slice(0,256).map(m=>m.loss)),last=mean(metrics.slice(-256).map(m=>m.loss));
    const utilization={samples:window.length,mean:mean(sorted),p10:sorted[Math.floor(sorted.length*.1)],
      minimum:sorted[0],maximum:sorted.at(-1),mean_power_w:mean(window.map(s=>s.power_w)),peak_device_memory_mib:Math.max(...window.map(s=>s.memory_mib))};
    if(!health.finite||!health.restored||!health.optimizer_restored||health.native_max_abs_error>2e-4||health.parameter_change<=0||last>=first)
      throw new Error('Training health gate failed');
    if(window.length<30||utilization.mean<90||utilization.p10<85)throw new Error('Sustained GPU utilization gate failed');
    write(healthPath,{...health,mean_first_256:first,mean_last_256:last,gpu:utilization});
    stamp('healthy',{health:'health.json',checkpoint:path.join('training',health.checkpoint)});
  }
  const candidates=[];let best=0,stalled=0;
  for(let target=config.stage_steps;target<=config.max_steps;target+=config.stage_steps){
    const health=await train(target), model=path.join(training,`step-${target}.blzn`);
    // Each fixed-budget stage is audited before committing more training time.
    const pilot=await audit('research/pilot.json','development',model,`pilot-${target}`);
    candidates.push({step:target,model,pilot});
    if(pilot.score>best){best=pilot.score;stalled=0;}else ++stalled;
    write(path.join(run,'progress.json'),{candidates,best_pilot_score:best,stalled,latest_training:health});
    if(stalled>=config.max_stalled_pilots)break;
  }
  candidates.sort((a,b)=>b.pilot.score-a.pilot.score);
  const viable=candidates.filter(c=>c.pilot.fallbacks<c.pilot.expected).slice(0,2);
  for(const candidate of viable)candidate.validation=await audit('research/corpus.json','validation',candidate.model,`validation-${candidate.step}`);
  viable.sort((a,b)=>b.validation.score-a.validation.score);
  const report={complete:true,release_approved:false,held_out_used:false,candidates,health:read(healthPath),timings:status.timings,
    reason:viable.length?'Selected using complete validation SCORE.':'Repeated development pilots returned only unreduced fallbacks; retain results for a new curriculum/decoder experiment.'};
  if(viable.length){fs.copyFileSync(viable[0].model,path.join(run,'selected.blzn'));report.selected_model='selected.blzn';report.model_sha256=hash(path.join(run,'selected.blzn'));}
  write(path.join(run,'report.json'),report);stamp(viable.length?'complete':'no-quality-signal',{report:'report.json'});
}catch(error){stamp(stopping||remaining()<60000?'incomplete':'failed',{error:String(error),release_approved:false});process.exitCode=1;}
finally{monitor.kill('SIGTERM');telemetry.end();}
