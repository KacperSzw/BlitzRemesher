#!/usr/bin/env node
// Local controller. All cloud mutations occur here or in its independent watchdog.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn,execFileSync} from 'node:child_process';
import {Api,Rental,apiKey,read,write,verifyPod,terminationDue,rentalDeadlines,retrySsh} from './runpod-api.mjs';
import {deployment} from './runpod-profile.mjs';
import {actionBudget,actionAccrued,continuationBudget,refactorBudget,pretrainingBudget,hardwareValidationBudget,pipelineValidationBudget} from './action-budget.mjs';
import {freshConditions} from './action-curriculum.mjs';

const [command,directory]=process.argv.slice(2);
if(!directory)throw new Error('runpod.mjs {prepare-gpu-refactor|prepare|prepare-actions|prepare-action-pilot|prepare-action-evaluation|prepare-action-staged|launch|status|stop|control|watchdog} RUN_DIRECTORY [EARLIER_DEADLINE_MS]');
const dir=path.resolve(directory),root=process.cwd(),statePath=dir+'/rental.json';
const refactor=command==='prepare-gpu-refactor';
const hardware=command==='prepare-hardware-validation';
const pipeline=command==='prepare-pipeline-validation';
const keyFile=path.join(os.homedir(),'.config/blitz/runpod-api-key');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const sh=s=>"'"+String(s).replaceAll("'","'\\''")+"'";
const sync=(cmd,args,options={})=>execFileSync(cmd,args,{encoding:'utf8',...options}).trim();
async function sha(file){const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(file))hash.update(chunk);return hash.digest('hex');}
function relative(file){if(path.isAbsolute(file)||file.split('/').includes('..')||/[\r\n\0]/.test(file)||file.startsWith('-'))throw new Error('Unsafe bundle path');return file;}
async function prepare(){
  if(fs.existsSync(dir+'/prepared.json'))throw new Error('Use a new directory for a new bundle');
  if(sync('git',['status','--porcelain','--untracked-files=normal']))throw new Error('Commit changes before preparing the immutable rental bundle');
  const revision=sync('git',['rev-parse','HEAD']),branch=sync('git',['branch','--show-current']);
  // The new deployment ships the complete immutable local commit in a bundle.
  if(!refactor&&!hardware&&!pipeline){const remote=sync('git',['ls-remote','origin','refs/heads/'+branch]).split(/\s/)[0];
    if(remote!==revision)throw new Error('Current branch is not fully pushed');}
  if(refactor&&deployment.id!=='gpu-refactor'&&!deployment.id.startsWith('coverage-pretrain-'))throw new Error('Select BLITZ_RUNPOD_PROFILE=gpu-refactor');
  if(hardware&&!['hardware-validation','hardware-validation-small','hardware-validation-ada','hardware-validation-ada16','hardware-validation-l40s'].includes(deployment.id))throw new Error('Select a hardware-validation RunPod profile');
  if(pipeline&&!['hardware-validation-l40s','pipeline-validation-l40s'].includes(deployment.id))throw new Error('Select a bounded L40S validation profile');
  fs.mkdirSync(dir,{recursive:true,mode:0o700});
  const stage=dir+'/input';fs.mkdirSync(stage,{recursive:true});
  sync('git',['bundle','create',stage+'/source.bundle','HEAD']);
  const files=[];
  const copy=async(source,destination,expected)=>{
    relative(destination);const actual=await sha(source);if(expected&&actual!==expected)throw new Error('Input checksum mismatch: '+source);
    const target=stage+'/'+destination;fs.mkdirSync(path.dirname(target),{recursive:true});
    fs.copyFileSync(source,target,fs.constants.COPYFILE_FICLONE);
    files.push({path:destination,sha256:actual,bytes:fs.statSync(target).size});
  };
  const evaluation=command==='prepare-action-evaluation';
  const actions=refactor||hardware||pipeline||['prepare-actions','prepare-action-pilot','prepare-action-evaluation','prepare-action-staged'].includes(command);
  if(!actions){const dataset=root+'/runs/neural/first-pass/data',index=read(dataset+'/index.json');
  if(!index.complete||index.assets.length!==66||await sha(dataset+'/index.json')!=='4faabd8f7411e6dfbc1fd0357e143f6627a6248e73b5c2421a947396cad1e5d7')
    throw new Error('Expected the approved original 66-asset teacher dataset');
  await copy(dataset+'/index.json','dataset/index.json');
  for(const asset of index.assets)await copy(dataset+'/'+relative(asset.path),'dataset/'+asset.path,asset.sha256);
  }
  const auditFiles=new Map();
  for(const [manifest,split] of [['research/pilot.json','development'],['research/corpus.json','validation']]){
    if(actions&&split==='validation')continue;
    for(const asset of read(manifest).assets.filter(a=>a.split===split))for(const file of asset.files)auditFiles.set(file.path,file.sha256);
  }
  if(actions&&!evaluation){const selected=new Set(['ph_sweet_potato','ph_painted_wooden_bench']);
    if(command==='prepare-action-staged'||refactor||hardware)for(const c of freshConditions)selected.add(c.asset);
    if(refactor||hardware||pipeline)for(const a of read(root+'/research/neural/prepared-pilot/selection.json').assets)selected.add(a.id);
    const allowed=new Set(read(root+'/research/neural/training-manifest.json').assets.map(a=>a.id));
    for(const asset of read(root+'/research/corpus.json').assets.filter(a=>selected.has(a.id))){if(!allowed.has(asset.id))throw new Error('Action proof asset outside training selection');for(const file of asset.files)auditFiles.set(file.path,file.sha256);selected.delete(asset.id);}
    if(selected.size)throw new Error('Missing action proof assets');
  }
  if(refactor){
    if(read(root+'/research/neural/next-training.json').launch!==true)throw new Error('Next training is not authorized');
    for(const name of ['training','validation'])for(const asset of read(root+'/research/neural/corpus-v2/'+name+'.json').assets)for(const file of asset.files)auditFiles.set(file.path,file.sha256);
    await copy(root+'/runs/neural/runpod-gpu-refactor-01/final-model.blzn','initial-model.blzn','9152bb42cb807a2e91fe3217ab6dc3bbcf11be12618bcd706acf71a8e3fff185');
    const source=process.env.BLITZ_REPLAY_CYCLE;if(!source)throw new Error('Previous collected cycle is required for the model comparison');const journal=read(source+'/latest.json');
    await copy(source+'/'+relative(journal.checkpoint)+'/model.blzn','previous-model.blzn');
  }
  if(pipeline){
    const source=process.env.BLITZ_REPLAY_CYCLE;if(!source)throw new Error('BLITZ_REPLAY_CYCLE must identify the collected run');
    const latest=read(source+'/latest.json');
    for(const name of ['latest.json','contract.json'])await copy(source+'/'+name,'forensic/'+name);
    for(const name of fs.readdirSync(source+'/'+relative(latest.checkpoint)))await copy(source+'/'+latest.checkpoint+'/'+relative(name),'forensic/'+latest.checkpoint+'/'+name);
    for(const shard of latest.datasets.slice(latest.resident_start??0))for(const name of ['actions.bin','index.json','contract.json'])await copy(source+'/data/'+relative(shard)+'/'+name,'forensic/data/'+shard+'/'+name);
    const config={mode:process.env.BLITZ_VALIDATION_MODE??'forensic',next_training_launch:false};
    if(['pipeline','packed-domain'].includes(config.mode)){
      for(const name of ['training','validation'])for(const asset of read(root+'/research/neural/corpus-v2/'+name+'.json').assets)for(const file of asset.files)auditFiles.set(file.path,file.sha256);
      await copy(root+'/runs/neural/runpod-gpu-refactor-01/final-model.blzn','initial-model.blzn','9152bb42cb807a2e91fe3217ab6dc3bbcf11be12618bcd706acf71a8e3fff185');
    }
    if(!['forensic','pipeline','packed-domain'].includes(config.mode))throw new Error('Unknown validation mode');
    write(stage+'/validation.json',config);files.push({path:'validation.json',sha256:await sha(stage+'/validation.json'),bytes:fs.statSync(stage+'/validation.json').size});
  }
  for(const [file,checksum] of auditFiles)await copy(root+'/'+relative(file),'assets/'+file,checksum);
  files.push({path:'source.bundle',sha256:await sha(stage+'/source.bundle'),bytes:fs.statSync(stage+'/source.bundle').size});
  write(stage+'/inputs.json',{revision,files});
  fs.writeFileSync(stage+'/inputs.sha256',files.map(f=>`${f.sha256}  ${f.path}`).join('\n')+'\n');
  sync('tar',['-cf',dir+'/input.tar','-C',stage,'.']);
  fs.mkdirSync(dir+'/control',{recursive:true});
  for(const name of ['runpod.mjs','runpod-api.mjs','runpod-profile.mjs','action-budget.mjs','action-curriculum.mjs'])fs.copyFileSync(root+'/research/neural/'+name,dir+'/control/'+name);
  write(dir+'/prepared.json',{revision,branch,deployment,experiment:pipeline?'pipeline-validation':hardware?'hardware-validation':refactor?'gpu-refactor':command==='prepare-action-staged'?'action-v2-staged':evaluation?'action-v2-evaluate':command==='prepare-action-pilot'?'action-v2-pilot':actions?'action-v2':'vertex-v1',archive_sha256:await sha(dir+'/input.tar'),archive_bytes:fs.statSync(dir+'/input.tar').size,files:files.length});
  fs.rmSync(stage,{recursive:true});
  console.log('Prepared checksummed '+(evaluation?'evaluation source, saved models and development pilot':actions?'action proof source/training meshes and development pilot':'source, original training shards, pilot and validation assets')+': '+dir);
}
function unitString(s){return '"'+s.replaceAll('\\','\\\\').replaceAll('"','\\"').replaceAll('%','%%')+'"';}
function installService(name,mode){
  const units=path.join(os.homedir(),'.config/systemd/user');fs.mkdirSync(units,{recursive:true});
  fs.writeFileSync(units+'/'+name+'.service',`[Unit]\nDescription=Bounded Blitz GPU ${mode}\n[Service]\nType=simple\nExecStart=${[process.execPath,dir+'/control/runpod.mjs',mode,dir].map(unitString).join(' ')}\nEnvironment=${unitString('PATH='+process.env.PATH)}\nEnvironment=${unitString('BLITZ_RUNPOD_PROFILE='+deployment.id)}\nRestart=on-failure\nRestartSec=10\nTimeoutStopSec=20\n[Install]\nWantedBy=default.target\n`);
}
async function launch(){
  if(fs.existsSync(statePath))throw new Error('This rental already has durable state; services resume it without creating another Pod');
  const prepared=read(dir+'/prepared.json');
  if(JSON.stringify(prepared.deployment)!==JSON.stringify(deployment))throw new Error('Prepared GPU profile differs; prepare a new bundle');
  if(await sha(dir+'/input.tar')!==prepared.archive_sha256)throw new Error('Prepared archive changed');
  const api=new Api(apiKey(keyFile)),quote=await api.quote(process.env.BLITZ_RUNPOD_DATA_CENTER);
  let budget;
  if(['action-v2','action-v2-pilot','action-v2-evaluate','action-v2-staged','gpu-refactor','hardware-validation','pipeline-validation'].includes(prepared.experiment)){
    const [pods,volumes,bill]=await Promise.all([api.pods(),api.request('GET','/network-volumes'),api.request('GET','/billing')]);
    if(pods.length||volumes.networkVolumes.length)throw new Error('Reconcile existing cloud resources before the bounded action experiment');
    const ledger=fs.readdirSync(root+'/runs/neural',{withFileTypes:true}).filter(e=>e.isDirectory()).map(e=>root+'/runs/neural/'+e.name+'/rental.json').filter(f=>fs.existsSync(f)).map(read);
    const continuation=prepared.experiment==='action-v2-staged';
    budget=(prepared.experiment==='pipeline-validation'?pipelineValidationBudget:prepared.experiment==='hardware-validation'?hardwareValidationBudget:prepared.experiment==='gpu-refactor'?pretrainingBudget:continuation?continuationBudget:actionBudget)({billed:bill.metadata.totals.totalAmount,additionalAccrued:actionAccrued(ledger),rate:deployment.gpu_hourly_usd_cap,minutes:prepared.experiment==='pipeline-validation'?deployment.setup_minutes+deployment.training_minutes+deployment.collection_minutes:prepared.experiment==='hardware-validation'?35:prepared.experiment==='gpu-refactor'?180:continuation?deployment.staged_rental_minutes:prepared.experiment==='action-v2'?60:90});write(dir+'/billing-before.json',bill);
  }
  sync('systemctl',['--user','show-environment']);
  sync('ssh-keygen',['-q','-t','ed25519','-N','','-f',dir+'/identity']);
  const name='blitz-'+crypto.randomUUID(),started=Date.now();
  const requested=process.argv[4]===undefined?undefined:Number(process.argv[4]);
  const deadlines=rentalDeadlines(started,budget?Math.min(requested??Infinity,started+budget.minutes*60000):requested);
  if(budget)deadlines.training_minutes=prepared.experiment==='pipeline-validation'?deployment.training_minutes:prepared.experiment==='hardware-validation'?10:prepared.experiment==='gpu-refactor'?150:prepared.experiment==='action-v2-staged'?deployment.staged_experiment_minutes:prepared.experiment==='action-v2'?20:50;
  write(statePath,{name,quote,deployment,experiment:prepared.experiment,budget,revision:prepared.revision,...deadlines});
  installService(name+'-watchdog','watchdog');installService(name+'-control','control');
  sync('systemctl',['--user','daemon-reload']);
  sync('systemctl',['--user','enable','--now',name+'-watchdog.service']);
  sync('systemctl',['--user','is-active',name+'-watchdog.service']);
  sync('systemctl',['--user','enable','--now',name+'-control.service']);
  console.log(JSON.stringify({name,quote,deadline:new Date(deadlines.deadline_ms).toISOString(),directory:dir}));
}
function sshArgs(endpoint){
  if(!endpoint||!/^[a-zA-Z0-9.:-]+$/.test(endpoint.host)||!/^\w+$/.test(endpoint.username)||!Number.isInteger(endpoint.port)||endpoint.port<1||endpoint.port>65535)
    throw new Error('Invalid direct SSH endpoint');
  return ['-i',dir+'/identity','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=accept-new',
    '-o','UserKnownHostsFile='+dir+'/known_hosts','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=6',
    '-p',String(endpoint.port),endpoint.username+'@'+endpoint.host];
}
async function remote(endpoint,command,{input,output,timeout=60000}={}){
  const fd=output?fs.openSync(output,'w'):undefined;
  const child=spawn('ssh',[...sshArgs(endpoint),command],{stdio:['pipe',fd??'pipe','pipe']});
  let data='',diagnostic='',timedOut=false;
  child.stdout?.on('data',chunk=>{data+=chunk;if(data.length>1024*1024)child.kill('SIGKILL');});
  child.stderr.on('data',chunk=>{diagnostic=(diagnostic+chunk).slice(-8192);});
  if(input){const stream=fs.createReadStream(input);stream.on('error',()=>child.kill('SIGKILL'));stream.pipe(child.stdin);child.once('close',()=>stream.destroy());}
  else child.stdin.end();
  child.stdin.on('error',()=>{});
  const timer=setTimeout(()=>{timedOut=true;child.kill('SIGKILL');},timeout);
  try{await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>{
    if(code===0)return resolve();
    // Private diagnostic file, never echoed to journals or sent to the provider.
    fs.appendFileSync(dir+'/transport-errors.jsonl',JSON.stringify({at:Date.now(),code,signal,timed_out:timedOut,stderr:diagnostic})+'\n',{mode:0o600});
    reject(Object.assign(new Error(`SSH failed (exit ${code}, signal ${signal}, timeout ${timedOut}); see transport-errors.jsonl`),{ssh_exit:code,ssh_timeout:timedOut}));
  });});return data.trim();}
  finally{clearTimeout(timer);if(fd!==undefined)fs.closeSync(fd);}
}
async function collect(endpoint,deadline){
  const left=()=>Math.max(1,deadline-Date.now());
  const checksum=(await retrySsh(()=>remote(endpoint,'cat /workspace/results.tar.gz.sha256',{timeout:Math.min(15000,left())}))).split(/\s/)[0];
  if(!/^[a-f0-9]{64}$/.test(checksum))throw new Error('Remote result checksum missing');
  await retrySsh(()=>remote(endpoint,'cat /workspace/results.tar.gz',{output:dir+'/results.tar.gz.part',timeout:left()}));
  if(await sha(dir+'/results.tar.gz.part')!==checksum)throw new Error('Downloaded results checksum mismatch');
  fs.renameSync(dir+'/results.tar.gz.part',dir+'/results.tar.gz');
  write(dir+'/collection.json',{verified:true,sha256:checksum,at:Date.now()});
}
async function control(){
  const api=new Api(apiKey(keyFile)),rental=new Rental(api,read(statePath),s=>write(statePath,s));
  const s=rental.state;
  if(s.compute_terminated){await rental.cleanupVolume(fs.existsSync(dir+'/collection.json')&&read(dir+'/collection.json').verified);return;}
  try{
    if(Date.now()>=s.deadline_ms||fs.existsSync(dir+'/stop-requested'))throw new Error('Rental deadline/cancellation reached');
    sync('systemctl',['--user','is-active',s.name+'-watchdog.service']);
    rental.commit({phase:'provisioning'});
    await rental.volume();await rental.pod(fs.readFileSync(dir+'/identity.pub','utf8'));
    let endpoint=s.endpoint;
    while(!endpoint&&Date.now()<s.setup_deadline_ms){
      const pod=await api.request('GET','/pods/'+encodeURIComponent(s.pod_id));
      if(!pod||['ERROR','EXITED','TERMINATED'].includes(pod.status))throw new Error('Pod unavailable during setup');
      if(pod.status==='RUNNING'&&pod.ssh?.direct){verifyPod(pod);endpoint=pod.ssh.direct;break;}
      await sleep(10000);
    }
    if(!endpoint)throw new Error('Setup deadline reached before SSH became available');
    rental.commit({endpoint,phase:'ssh'});
    for(;;){
      try{await remote(endpoint,'true',{timeout:15000});break;}
      catch(error){if(Date.now()+15000>=s.setup_deadline_ms)throw error;await sleep(5000);}
    }
    if(!s.uploaded){
      const prepared=read(dir+'/prepared.json');
      rental.commit({phase:'upload'});
      // Restarting a failed upload is repeatable. The lock prevents overlapping
      // remote cats after transport loss; only a completed stream is promoted.
      await retrySsh(()=>remote(endpoint,"flock /workspace/upload.lock sh -c 'cat > /workspace/input.tar.part && mv /workspace/input.tar.part /workspace/input.tar'",{input:dir+'/input.tar',timeout:Math.max(1,s.setup_deadline_ms-Date.now())}));
      rental.commit({phase:'verify-upload'});
      const actual=(await retrySsh(()=>remote(endpoint,'sha256sum /workspace/input.tar'))).split(/\s/)[0];
      if(actual!==prepared.archive_sha256)throw new Error('Uploaded input checksum mismatch');
      rental.commit({phase:'restore'});
      await remote(endpoint,'cd /workspace && tar --no-same-owner --no-same-permissions -xf input.tar && sha256sum --quiet --check inputs.sha256 && if [ ! -d project/.git ]; then git clone source.bundle project; fi && cd project && git checkout '+sh(prepared.revision)+' && cp -r /workspace/assets/data/. data/ && rm /workspace/input.tar', {timeout:Math.max(1,s.setup_deadline_ms-Date.now())});
      rental.commit({uploaded:true});
    }
    rental.commit({phase:'start-job',job_requested:true});
    // A remote marker survives SSH loss and controller restarts. Never start twice.
    await retrySsh(()=>remote(endpoint,'flock -o /workspace/launch.lock bash -c '+sh('if [ ! -f /workspace/job-started ]; then touch /workspace/job-started; nohup env BLITZ_RUNPOD_PROFILE='+sh(deployment.id)+' bash /workspace/project/research/neural/cloud-job.sh '+s.setup_deadline_ms+' '+s.training_deadline_ms+' '+s.training_minutes+' '+sh(s.experiment??'vertex-v1')+' > /workspace/launch.log 2>&1 < /dev/null & fi')));
    while(Date.now()<s.deadline_ms-30000){
      if(fs.existsSync(dir+'/stop-requested'))throw new Error('Cancellation requested');
      const phase=await retrySsh(()=>remote(endpoint,'if [ -f /workspace/job-finished ]; then echo finished; elif [ -f /workspace/results/setup-complete.json ]; then echo training; else echo setup; fi'));
      if(phase==='training'&&!s.setup_complete){
        const timing=JSON.parse(await retrySsh(()=>remote(endpoint,'cat /workspace/results/setup-complete.json')));
        if(!Number.isFinite(timing.at)||!Number.isFinite(timing.training_deadline_ms)||timing.training_deadline_ms> s.training_deadline_ms)
          throw new Error('Invalid remote training deadline');
        rental.commit({setup_complete:true,phase:'training',training_started_at:timing.at,training_deadline_ms:timing.training_deadline_ms,
          deadline_ms:Math.min(s.deadline_ms,timing.training_deadline_ms+deployment.collection_minutes*60000)});
      }
      if(phase==='setup'&&s.phase!=='setup')rental.commit({phase:'setup'});
      if(phase==='finished'){rental.commit({phase:'collecting'});await collect(endpoint,s.deadline_ms-15000);break;}
      if(phase==='training'&&(s.experiment?.startsWith('action-v2')||s.experiment==='gpu-refactor')){
        const live=await retrySsh(()=>remote(endpoint,'if [ -f /workspace/results/phase.json ]; then cat /workspace/results/phase.json; else echo null; fi'));
        const p=JSON.parse(live);if(p&&['validation','preparation','training','audit','finished','coverage-contracts','matched-coverage-benchmark','coverage-curriculum-validation','expanded-training-preflight','resident-learning-cycle','final-validation-comparison'].includes(p.phase)&&p.phase!==s.phase)rental.commit({phase:p.phase});
      }
      if(!s.setup_complete&&Date.now()>=s.setup_deadline_ms)throw new Error('Setup exceeded its bounded deadline');
      await sleep(10000);
    }
  }catch(error){
    rental.commit({failure_phase:s.phase,error:String(error),...(error.detail?{error_detail:error.detail}:{})});console.error(String(error));
    // Preserve available evidence on a setup failure or lost SSH session. The
    // watchdog still owns the deadline while this best-effort collection runs.
    if(s.endpoint&&Date.now()<s.deadline_ms-30000){
      try{
        const finished=await remote(s.endpoint,'test -f /workspace/job-finished && echo finished',{timeout:10000});
        if(finished==='finished')await collect(s.endpoint,s.deadline_ms-15000);
      }catch{} // Failed collection leaves the network volume intact.
    }
  }
  finally{
    await rental.terminate();
    await rental.cleanupVolume(fs.existsSync(dir+'/collection.json')&&read(dir+'/collection.json').verified);
    rental.commit({phase:s.volume_deleted?'complete':'stopped_uncollected'});
    console.log(JSON.stringify({compute_terminated:true,volume_preserved:!s.volume_deleted,results_verified:fs.existsSync(dir+'/collection.json')}));
  }
}
async function watchdog(){
  const api=new Api(apiKey(keyFile));
  for(;;){
    const state=read(statePath),completed=state.compute_terminated;
    const expired=terminationDue(state,Date.now(),fs.existsSync(dir+'/stop-requested'));
    if(completed)return;
    if(expired){
      const rental=new Rental(api,state,s=>write(dir+'/watchdog.json',s));
      await rental.terminate();
      // Keep polling through the deadline for a delayed, ambiguous POST response.
      if(Date.now()>state.deadline_ms+60000)return;
    }
    await sleep(5000);
  }
}
if(['prepare-pipeline-validation','prepare-hardware-validation','prepare-gpu-refactor','prepare','prepare-actions','prepare-action-pilot','prepare-action-evaluation','prepare-action-staged'].includes(command))await prepare();
else if(command==='launch')await launch();
else if(command==='control')await control();
else if(command==='watchdog')await watchdog();
else if(command==='stop'){fs.writeFileSync(dir+'/stop-requested','Stop requested\n',{mode:0o600});console.log('Termination requested; the watchdog will preserve uncollected storage.');}
else if(command==='status'){
  for(const name of ['rental.json','watchdog.json','collection.json'])if(fs.existsSync(dir+'/'+name))console.log(name,JSON.stringify(read(dir+'/'+name),null,2));
}
else throw new Error('Unknown command: '+command);
