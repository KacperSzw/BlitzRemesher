// Matched full development comparison; never promotes an incomplete batch.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {read,write} from './runpod-api.mjs';
import {pilotDecision} from './action-gates.mjs';
const [directory,...args]=process.argv.slice(2),models=args.slice(0,3);
if(!directory||models.length!==3)throw new Error('action-pilot.mjs OUTPUT MODEL1 MODEL2 MODEL3 [--previous REPORT] [--minutes 45] [--action-trials 8] [--action-batch 32]');
let previousFile,minutes=45,methodMinutes=5,trials=8,batch=32,memory=6144,scenario='pilot',confirmation='cpu';
for(let i=3;i<args.length;i+=2){if(args[i]==='--previous')previousFile=args[i+1];else if(args[i]==='--minutes')minutes=Number(args[i+1]);else if(args[i]==='--method-minutes')methodMinutes=Number(args[i+1]);else if(args[i]==='--action-trials')trials=Number(args[i+1]);else if(args[i]==='--action-batch')batch=Number(args[i+1]);else if(args[i]==='--gpu-memory-mib')memory=Number(args[i+1]);else if(args[i]==='--scenario')scenario=args[i+1];else if(args[i]==='--neural-confirmation')confirmation=args[i+1];else throw new Error('Unknown pilot option');}
if(!['pilot','diagnostic'].includes(scenario)||!['cpu','gpu','compare'].includes(confirmation)||previousFile&&scenario!=='pilot')throw new Error('Invalid comparison scenario or confirmation backend');
if(!Number.isFinite(minutes)||minutes<=0||minutes>50||!Number.isInteger(trials)||trials<1||trials>65536||!Number.isInteger(batch)||batch<1||batch>64||!Number.isInteger(memory)||memory<128||memory>65536)throw new Error('Invalid bounded pilot settings');
if(!Number.isFinite(methodMinutes)||methodMinutes<=0||methodMinutes>50)throw new Error('Invalid method time limit');
const dir=path.resolve(directory);fs.mkdirSync(dir,{recursive:true});const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest=scenario==='pilot'?'research/pilot.json':'research/neural/action-diagnostic.json',config='research/neural/action-pilot.json';
const contract={manifest:hash(manifest),config:hash(config),protocol:hash('research/PROTOCOL.md'),binary:hash('build/neural/blitz'),trials,batch,memory,confirmation,scenario,methodMinutes,models:models.map(hash)};
if(new Set(contract.models).size!==3)throw new Error('Three independently seeded model files are required');
if(fs.existsSync(dir+'/contract.json')&&JSON.stringify(read(dir+'/contract.json'))!==JSON.stringify(contract))throw new Error('Pilot inputs changed; choose another output directory');write(dir+'/contract.json',contract);
const comparison={manifest:contract.manifest,config:contract.config,protocol:contract.protocol,binary:contract.binary,trials,batch,memory,confirmation,scenario};
let active,cancelled=false;for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{cancelled=true;active?.kill('SIGTERM');});
const deadline=Date.now()+minutes*60000,runs=[];
const methods=[...models.map((model,i)=>({name:'learned-'+i,ranking:'learned',model,seed:i+1})),
  {name:'constant',ranking:'constant',model:models[0],seed:1},
  ...models.map((model,i)=>({name:'shuffled-'+i,ranking:'shuffled',model,seed:i+1})),
  {name:'shortest',ranking:'shortest',model:models[0],seed:1},{name:'current-plane',ranking:'current-plane',model:models[0],seed:1}];
for(const method of methods){if(cancelled||Date.now()+15000>=deadline)break;
  const out=dir+'/'+method.name,batchMinutes=Math.min(methodMinutes,(deadline-Date.now()-10000)/60000),fd=fs.openSync(dir+'/'+method.name+'.log','a');
  write(dir+'/progress.json',{phase:'audit',scenario,method:method.name,completed_methods:runs.length,at:Date.now()});
  const began=Date.now();
  active=spawn('build/neural/blitz',['bench',manifest,config,out,'--neural-model',method.model,'--neural-control',method.ranking,'--ranking-seed',String(method.seed),'--action-trials',String(trials),'--action-batch',String(batch),'--gpu-memory-mib',String(memory),'--neural-confirmation',confirmation,'--minutes',String(batchMinutes)],{stdio:['ignore',fd,fd]});
  let hard;const timer=setTimeout(()=>{active?.kill('SIGTERM');hard=setTimeout(()=>active?.kill('SIGKILL'),5000);},Math.min(deadline-Date.now(),(batchMinutes*60+5)*1000));
  let code;try{code=await new Promise((resolve,reject)=>{active.once('error',reject);active.once('close',resolve);});}finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);active=undefined;}
  const summary=fs.existsSync(out+'/summary.json')?read(out+'/summary.json'):null;
  const rows=read(manifest).assets.map(a=>out+'/rows/'+a.id+'.json').filter(f=>fs.existsSync(f)).map(read);
  const health_complete=rows.every(r=>!r.failed&&!r.neural?.resource_failures&&!r.neural?.confirmation_resources&&!r.neural?.confirmation_nonfinite&&!r.neural?.confirmation_disagreements);
  runs.push({...method,code,summary,health_complete,directory:out,wall_seconds:(Date.now()-began)/1000,rows:rows.map(r=>({id:r.id,category:r.category,complete:r.complete,ratio:r.ratio,seconds:r.seconds,neural:r.neural}))});
  write(dir+'/report.json',{complete:false,comparison,runs,gate:{passed:false,reason:'incomplete'}});
  if(code!==0||!summary?.complete||!health_complete)break;
}
const gate=scenario==='pilot'?pilotDecision(runs):{passed:false,reason:'diagnostic_only'},previous=previousFile?read(previousFile):null;
const persisted=gate.passed&&previous?.gate?.passed&&JSON.stringify(previous.comparison)===JSON.stringify(comparison)&&previous.model_hashes?.length===3&&previous.model_hashes.every((old,i)=>old!==contract.models[i]);
write(dir+'/report.json',{complete:runs.length===methods.length&&runs.every(r=>r.summary?.complete&&r.health_complete&&r.code===0),comparison,model_hashes:contract.models,runs,gate,persisted:!!persisted,release_quality_proven:false});
