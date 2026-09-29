// Matched full development comparison; never promotes an incomplete batch.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {read,write} from './runpod-api.mjs';
import {pilotDecision} from './action-gates.mjs';
const [directory,...args]=process.argv.slice(2),models=args.slice(0,3);
if(!directory||models.length!==3)throw new Error('action-pilot.mjs OUTPUT MODEL1 MODEL2 MODEL3 [--previous REPORT] [--minutes 45] [--action-trials 64]');
let previousFile,minutes=45,trials=64;
for(let i=3;i<args.length;i+=2){if(args[i]==='--previous')previousFile=args[i+1];else if(args[i]==='--minutes')minutes=Number(args[i+1]);else if(args[i]==='--action-trials')trials=Number(args[i+1]);else throw new Error('Unknown pilot option');}
if(!Number.isFinite(minutes)||minutes<=0||minutes>50||!Number.isInteger(trials)||trials<1||trials>65536)throw new Error('Invalid bounded pilot settings');
const dir=path.resolve(directory);fs.mkdirSync(dir,{recursive:true});const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest='research/pilot.json',config='research/neural/action-pilot.json';
const contract={manifest:hash(manifest),config:hash(config),protocol:hash('research/PROTOCOL.md'),binary:hash('build/neural/blitz'),trials,models:models.map(hash)};
if(new Set(contract.models).size!==3)throw new Error('Three independently seeded model files are required');
if(fs.existsSync(dir+'/contract.json')&&JSON.stringify(read(dir+'/contract.json'))!==JSON.stringify(contract))throw new Error('Pilot inputs changed; choose another output directory');write(dir+'/contract.json',contract);
const comparison={manifest:contract.manifest,config:contract.config,protocol:contract.protocol,binary:contract.binary,trials};
let active,cancelled=false;for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{cancelled=true;active?.kill('SIGTERM');});
const deadline=Date.now()+minutes*60000,runs=[];
const methods=[...models.map((model,i)=>({name:'learned-'+i,ranking:'learned',model,seed:i+1})),
  {name:'constant',ranking:'constant',model:models[0],seed:1},
  ...models.map((model,i)=>({name:'shuffled-'+i,ranking:'shuffled',model,seed:i+1})),
  {name:'shortest',ranking:'shortest',model:models[0],seed:1},{name:'current-plane',ranking:'current-plane',model:models[0],seed:1}];
for(const method of methods){if(cancelled||Date.now()+15000>=deadline)break;
  const out=dir+'/'+method.name,batchMinutes=Math.min(5,(deadline-Date.now()-10000)/60000),fd=fs.openSync(dir+'/'+method.name+'.log','a');
  active=spawn('build/neural/blitz',['bench',manifest,config,out,'--neural-model',method.model,'--neural-control',method.ranking,'--ranking-seed',String(method.seed),'--action-trials',String(trials),'--minutes',String(batchMinutes)],{stdio:['ignore',fd,fd]});
  let hard;const timer=setTimeout(()=>{active?.kill('SIGTERM');hard=setTimeout(()=>active?.kill('SIGKILL'),5000);},Math.min(deadline-Date.now(),(batchMinutes*60+5)*1000));
  let code;try{code=await new Promise((resolve,reject)=>{active.once('error',reject);active.once('close',resolve);});}finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);active=undefined;}
  const summary=fs.existsSync(out+'/summary.json')?read(out+'/summary.json'):null;runs.push({...method,code,summary,directory:out});
  write(dir+'/report.json',{complete:false,comparison,runs,gate:{passed:false,reason:'incomplete'}});
  if(code!==0||!summary?.complete)break;
}
const gate=pilotDecision(runs),previous=previousFile?read(previousFile):null;
const persisted=gate.passed&&previous?.gate?.passed&&JSON.stringify(previous.comparison)===JSON.stringify(comparison)&&previous.model_hashes?.length===3&&previous.model_hashes.every((old,i)=>old!==contract.models[i]);
write(dir+'/report.json',{complete:runs.length===methods.length&&runs.every(r=>r.summary?.complete),comparison,model_hashes:contract.models,runs,gate,persisted:!!persisted,release_quality_proven:false});
