// Remote bounded action proof. No account credential is available here.
import fs from 'node:fs';
import {spawn} from 'node:child_process';
import {read,write} from './runpod-api.mjs';
import {endpointDecision} from './action-gates.mjs';
import {generalize} from './action-generalize.mjs';
const [setup,latest,minutes]=process.argv.slice(2,5).map(Number),experiment=process.argv[5]??'action-v2';
const pilot=experiment==='action-v2-pilot';
if(!['action-v2','action-v2-pilot'].includes(experiment)||![setup,latest,minutes].every(Number.isFinite)||Date.now()>=setup||minutes<=0||minutes>(pilot?50:20))throw new Error('Invalid action experiment deadline');
const started=Date.now(),deadline=Math.min(latest,started+minutes*60000),results='/workspace/results',root=results+'/'+experiment;
fs.mkdirSync(root,{recursive:true});write(results+'/setup-complete.json',{at:started,training_minutes:minutes,training_deadline_ms:deadline});
let active,cancelled=false,phase='preparation';
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{cancelled=true;if(active)process.kill(-active.pid,'SIGTERM');});
const monitor=spawn('nvidia-smi',['--query-gpu=utilization.gpu,power.draw,memory.used','--format=csv,noheader,nounits','-l','1']);
const telemetry=fs.createWriteStream(root+'/gpu.jsonl');let pending='',monitorError;
monitor.on('error',e=>{monitorError=String(e);});monitor.stdout.on('data',chunk=>{pending+=chunk;let at;while((at=pending.indexOf('\n'))>=0){const values=pending.slice(0,at).split(',').map(Number);pending=pending.slice(at+1);if(values.length===3&&values.every(Number.isFinite))telemetry.write(JSON.stringify({at:Date.now(),phase,gpu:values[0],power_w:values[1],memory_mib:values[2]})+'\n');}});
async function execute(name,args,log,maximumMinutes){
  if(cancelled||Date.now()+15000>=deadline)throw new Error('Action experiment deadline/cancellation');
  const fd=fs.openSync(log,'a'),end=Math.min(deadline-10000,Date.now()+maximumMinutes*60000);let hard,timedOut=false,memoryExceeded=false;
  active=spawn(name==='node'?process.execPath:'build/neural/'+name,args,{stdio:['ignore',fd,fd],detached:true});
  const stop=()=>{if(active){process.kill(-active.pid,'SIGTERM');hard=setTimeout(()=>{if(active)process.kill(-active.pid,'SIGKILL');},5000);}};
  const timer=setTimeout(()=>{timedOut=true;stop();},Math.max(1,end-Date.now()));
  const memory=setInterval(()=>{let kib=0;for(const pid of fs.readdirSync('/proc').filter(p=>/^\d+$/.test(p))){try{const stat=fs.readFileSync(`/proc/${pid}/stat`,'utf8'),fields=stat.slice(stat.lastIndexOf(')')+2).split(' ');if(Number(fields[2])!==active?.pid)continue;kib+=Number(fs.readFileSync(`/proc/${pid}/status`,'utf8').match(/^VmRSS:\s+(\d+)/m)?.[1]??0);}catch{}}
    if(kib>24*1024*1024){memoryExceeded=true;stop();}},1000);
  try{const code=await new Promise((resolve,reject)=>{active.once('error',reject);active.once('close',resolve);});if(memoryExceeded||timedOut||![0,2].includes(code))throw new Error(`${name} failed (${code}, deadline=${timedOut}, memory=${memoryExceeded})`);return code;}
  finally{clearTimeout(timer);clearTimeout(hard);clearInterval(memory);active=undefined;fs.closeSync(fd);}
}
const result={schema:2,started,deadline,complete:false,endpoint_gate_passed:false,seeds:[],quality_proven:false};
try{
  if(pilot){Object.assign(result,await generalize({root,execute,deadline,phase:value=>{phase=value;}}));}
  else {
  const data=root+'/proof-data';await execute('blitz-neural-action-prepare',['ph_sweet_potato',data,'--states','64','--pixels','32','--minutes','4'],root+'/prepare.log',4.5);
  if(!read(data+'/index.json').complete)throw new Error('Endpoint teacher preparation incomplete');
  for(const seed of [101,202,303]){
    const run=root+'/seed-'+seed,history=[];let decision={stop:false,passed:false};
    for(let stage=1;stage<=12&&!decision.stop;stage++){
      phase='training';await execute('blitz-neural-action-train',[data,run,'--steps',String(stage*4096),'--minutes','4','--batch','512','--seed',String(seed)],root+`/train-${seed}.log`,4.5);
      const health=read(run+'/latest.json');phase='audit';const report=root+`/proof-${seed}-${stage}.json`;
      await execute('blitz-neural-action-proof',[run+'/'+health.model,data,report,'--minutes','2'],root+`/proof-${seed}.log`,2.5);
      const proof=read(report);history.push({health,proof});decision=endpointDecision(history);write(root+`/history-${seed}.json`,history);
      write(root+'/progress.json',{seed,stage,decision,health,proof:report,at:Date.now()});
    }
    result.seeds.push({seed,...decision});write(root+'/result.json',result);if(!decision.passed){result.stop_reason=decision.reason??'endpoint_stage_limit';break;}
  }
  result.endpoint_gate_passed=result.seeds.length===3&&result.seeds.every(s=>s.passed);
  result.complete=true;result.next_phase=result.endpoint_gate_passed?'matched_development_generalization':'diagnose_endpoint_proof';
  }
  if(monitorError)throw new Error(monitorError);
}catch(error){result.error=String(error);process.exitCode=1;}
finally{result.finished=Date.now();write(root+'/result.json',result);write(results+'/job.json',{code:process.exitCode??0,experiment,result:root+'/result.json'});monitor.kill('SIGTERM');telemetry.end();}
