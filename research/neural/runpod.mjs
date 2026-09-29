#!/usr/bin/env node
// Local controller. All cloud mutations occur here or in its independent watchdog.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn,execFileSync} from 'node:child_process';
import {Api,Rental,apiKey,read,write,verifyPod,terminationDue,rentalDeadlines} from './runpod-api.mjs';
import {deployment} from './runpod-profile.mjs';

const [command,directory]=process.argv.slice(2);
if(!directory)throw new Error('runpod.mjs {prepare|launch|status|stop|control|watchdog} RUN_DIRECTORY [EARLIER_DEADLINE_MS]');
const dir=path.resolve(directory),root=process.cwd(),statePath=dir+'/rental.json';
const keyFile=path.join(os.homedir(),'.config/blitz/runpod-api-key');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const sh=s=>"'"+String(s).replaceAll("'","'\\''")+"'";
const sync=(cmd,args,options={})=>execFileSync(cmd,args,{encoding:'utf8',...options}).trim();
async function sha(file){const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(file))hash.update(chunk);return hash.digest('hex');}
function relative(file){if(path.isAbsolute(file)||file.split('/').includes('..')||/[\r\n\0]/.test(file)||file.startsWith('-'))throw new Error('Unsafe bundle path');return file;}
async function prepare(){
  if(fs.existsSync(dir+'/prepared.json'))throw new Error('Use a new directory for a new bundle');
  if(sync('git',['status','--porcelain','--untracked-files=normal']))throw new Error('Commit and push changes before preparing the rental');
  const revision=sync('git',['rev-parse','HEAD']),branch=sync('git',['branch','--show-current']);
  const remote=sync('git',['ls-remote','origin','refs/heads/'+branch]).split(/\s/)[0];
  if(remote!==revision)throw new Error('Current branch is not fully pushed');
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
  const dataset=root+'/runs/neural/first-pass/data',index=read(dataset+'/index.json');
  if(!index.complete||index.assets.length!==66||await sha(dataset+'/index.json')!=='4faabd8f7411e6dfbc1fd0357e143f6627a6248e73b5c2421a947396cad1e5d7')
    throw new Error('Expected the approved original 66-asset teacher dataset');
  await copy(dataset+'/index.json','dataset/index.json');
  for(const asset of index.assets)await copy(dataset+'/'+relative(asset.path),'dataset/'+asset.path,asset.sha256);
  const auditFiles=new Map();
  for(const [manifest,split] of [['research/pilot.json','development'],['research/corpus.json','validation']]){
    for(const asset of read(manifest).assets.filter(a=>a.split===split))for(const file of asset.files)auditFiles.set(file.path,file.sha256);
  }
  for(const [file,checksum] of auditFiles)await copy(root+'/'+relative(file),'assets/'+file,checksum);
  files.push({path:'source.bundle',sha256:await sha(stage+'/source.bundle'),bytes:fs.statSync(stage+'/source.bundle').size});
  write(stage+'/inputs.json',{revision,files});
  fs.writeFileSync(stage+'/inputs.sha256',files.map(f=>`${f.sha256}  ${f.path}`).join('\n')+'\n');
  sync('tar',['-cf',dir+'/input.tar','-C',stage,'.']);
  fs.mkdirSync(dir+'/control',{recursive:true});
  for(const name of ['runpod.mjs','runpod-api.mjs','runpod-profile.mjs'])fs.copyFileSync(root+'/research/neural/'+name,dir+'/control/'+name);
  write(dir+'/prepared.json',{revision,branch,deployment,archive_sha256:await sha(dir+'/input.tar'),archive_bytes:fs.statSync(dir+'/input.tar').size,files:files.length});
  fs.rmSync(stage,{recursive:true});
  console.log('Prepared checksummed source, original training shards, pilot and validation assets: '+dir);
}
function unitString(s){return '"'+s.replaceAll('\\','\\\\').replaceAll('"','\\"').replaceAll('%','%%')+'"';}
function installService(name,mode){
  const units=path.join(os.homedir(),'.config/systemd/user');fs.mkdirSync(units,{recursive:true});
  fs.writeFileSync(units+'/'+name+'.service',`[Unit]\nDescription=Bounded Blitz GPU ${mode}\n[Service]\nType=simple\nExecStart=${[process.execPath,dir+'/control/runpod.mjs',mode,dir].map(unitString).join(' ')}\nEnvironment=${unitString('PATH='+process.env.PATH)}\nRestart=on-failure\nRestartSec=10\nTimeoutStopSec=20\n[Install]\nWantedBy=default.target\n`);
}
async function launch(){
  if(fs.existsSync(statePath))throw new Error('This rental already has durable state; services resume it without creating another Pod');
  const prepared=read(dir+'/prepared.json');
  if(JSON.stringify(prepared.deployment)!==JSON.stringify(deployment))throw new Error('Prepared GPU profile differs; prepare a new bundle');
  if(await sha(dir+'/input.tar')!==prepared.archive_sha256)throw new Error('Prepared archive changed');
  const api=new Api(apiKey(keyFile)),quote=await api.quote(process.env.BLITZ_RUNPOD_DATA_CENTER);
  sync('systemctl',['--user','show-environment']);
  sync('ssh-keygen',['-q','-t','ed25519','-N','','-f',dir+'/identity']);
  const name='blitz-'+crypto.randomUUID(),started=Date.now();
  const deadlines=rentalDeadlines(started,process.argv[4]===undefined?undefined:Number(process.argv[4]));
  write(statePath,{name,quote,deployment,revision:prepared.revision,...deadlines});
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
    '-o','UserKnownHostsFile='+dir+'/known_hosts','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
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
    reject(new Error(`SSH failed (exit ${code}, signal ${signal}, timeout ${timedOut}); see transport-errors.jsonl`));
  });});return data.trim();}
  finally{clearTimeout(timer);if(fd!==undefined)fs.closeSync(fd);}
}
async function collect(endpoint,deadline){
  const left=()=>Math.max(1,deadline-Date.now());
  const checksum=(await remote(endpoint,'cat /workspace/results.tar.gz.sha256',{timeout:Math.min(15000,left())})).split(/\s/)[0];
  if(!/^[a-f0-9]{64}$/.test(checksum))throw new Error('Remote result checksum missing');
  await remote(endpoint,'cat /workspace/results.tar.gz',{output:dir+'/results.tar.gz.part',timeout:left()});
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
      await remote(endpoint,'cat > /workspace/input.tar',{input:dir+'/input.tar',timeout:Math.max(1,s.setup_deadline_ms-Date.now())});
      rental.commit({phase:'verify-upload'});
      const actual=(await remote(endpoint,'sha256sum /workspace/input.tar')).split(/\s/)[0];
      if(actual!==prepared.archive_sha256)throw new Error('Uploaded input checksum mismatch');
      rental.commit({phase:'restore'});
      await remote(endpoint,'cd /workspace && tar --no-same-owner --no-same-permissions -xf input.tar && sha256sum --quiet --check inputs.sha256 && if [ ! -d project/.git ]; then git clone source.bundle project; fi && cd project && git checkout '+sh(prepared.revision)+' && cp -r /workspace/assets/data/. data/ && rm /workspace/input.tar', {timeout:Math.max(1,s.setup_deadline_ms-Date.now())});
      rental.commit({uploaded:true});
    }
    rental.commit({phase:'start-job'});
    // A remote marker survives SSH loss and controller restarts. Never start twice.
    await remote(endpoint,'flock -o /workspace/launch.lock bash -c '+sh('if [ ! -f /workspace/job-started ]; then touch /workspace/job-started; nohup bash /workspace/project/research/neural/cloud-job.sh '+s.setup_deadline_ms+' '+s.training_deadline_ms+' '+s.training_minutes+' > /workspace/launch.log 2>&1 < /dev/null & fi'));
    while(Date.now()<s.deadline_ms-30000){
      if(fs.existsSync(dir+'/stop-requested'))throw new Error('Cancellation requested');
      const phase=await remote(endpoint,'if [ -f /workspace/job-finished ]; then echo finished; elif [ -f /workspace/results/setup-complete.json ]; then echo training; else echo setup; fi');
      if(phase==='training'&&!s.setup_complete){
        const timing=JSON.parse(await remote(endpoint,'cat /workspace/results/setup-complete.json'));
        if(!Number.isFinite(timing.at)||!Number.isFinite(timing.training_deadline_ms)||timing.training_deadline_ms> s.training_deadline_ms)
          throw new Error('Invalid remote training deadline');
        rental.commit({setup_complete:true,phase:'training',training_started_at:timing.at,training_deadline_ms:timing.training_deadline_ms,
          deadline_ms:Math.min(s.deadline_ms,timing.training_deadline_ms+deployment.collection_minutes*60000)});
      }
      if(phase==='setup'&&s.phase!=='setup')rental.commit({phase:'setup'});
      if(phase==='finished'){await collect(endpoint,s.deadline_ms-15000);break;}
      if(!s.setup_complete&&Date.now()>=s.setup_deadline_ms)throw new Error('Setup exceeded 30 minutes');
      await sleep(10000);
    }
  }catch(error){
    rental.commit({error:String(error)});console.error(String(error));
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
if(command==='prepare')await prepare();
else if(command==='launch')await launch();
else if(command==='control')await control();
else if(command==='watchdog')await watchdog();
else if(command==='stop'){fs.writeFileSync(dir+'/stop-requested','Stop requested\n',{mode:0o600});console.log('Termination requested; the watchdog will preserve uncollected storage.');}
else if(command==='status'){
  for(const name of ['rental.json','watchdog.json','collection.json'])if(fs.existsSync(dir+'/'+name))console.log(name,JSON.stringify(read(dir+'/'+name),null,2));
}
else throw new Error('Unknown command: '+command);
