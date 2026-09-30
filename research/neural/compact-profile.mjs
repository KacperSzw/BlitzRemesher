#!/usr/bin/env node
// Alternating local comparisons. FP32 controls isolate scheduling changes;
// compact records are measured separately because they change training inputs.
import fs from 'node:fs';
import path from 'node:path';
import {spawn,execFileSync} from 'node:child_process';
import {hash,modelPayload} from './refactor-cycle.mjs';
const [directory,model,checkpoint,repeatsText='3']=process.argv.slice(2),repeats=Number(repeatsText);
if(!directory||!model||!checkpoint||!Number.isInteger(repeats)||repeats<1||repeats>5||fs.existsSync(directory))throw Error('compact-profile.mjs FRESH_DIRECTORY MODEL CHECKPOINT [REPEATS 1..5]');
const root=path.resolve(directory);fs.mkdirSync(root,{recursive:true});
const baseline='runs/neural/compact-pipeline/baseline-bin/blitz-neural-cycle',current='build/neural/blitz-neural-cycle';
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const gpu=()=>execFileSync('nvidia-smi',['--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],{encoding:'utf8'}).trim();
const report={complete:false,score:null,started:new Date().toISOString(),baseline_sha256:hash(baseline),current_sha256:hash(current),model_sha256:hash(model),checkpoint_sha256:hash(checkpoint),protocol_sha256:hash('research/PROTOCOL.md'),rows:[]};
const save=()=>fs.writeFileSync(root+'/comparison.json',JSON.stringify(report,null,2)+'\n');save();
async function run(kind,repeat){
 const directory=root+'/'+kind+'-'+repeat,binary=kind==='baseline'?baseline:current;
 const args=[directory,'--states','8','--updates','2048','--minutes','3','--gpu-memory-mib','1024','--vertex-storage','fp32','--initialize',path.resolve(model),'--warmstart',path.resolve(checkpoint),...(kind==='baseline'?[]:['--data-storage',kind==='compact'?'compact':'fp32'])];
 const row={kind,repeat,binary,args,gpu_samples:[gpu()]};report.rows.push(row);save();const fd=fs.openSync(directory+'.log','w'),start=performance.now(),child=spawn(binary,args,{stdio:['ignore',fd,fd]});
 const telemetry=setInterval(()=>row.gpu_samples.push(gpu()),1000);const stop=setTimeout(()=>child.kill('SIGTERM'),230000);
 try{row.code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',resolve);});}finally{clearInterval(telemetry);clearTimeout(stop);row.seconds=(performance.now()-start)/1000;fs.closeSync(fd);}
 row.result=fs.existsSync(directory+'/report.json')?read(directory+'/report.json'):null;
 row.complete=row.code===0&&row.result?.complete&&row.result.states===54&&row.result.updates_this_invocation===12288;
 if(row.complete){row.data_hashes=row.result.datasets.map(name=>hash(directory+'/data/'+name+'/actions.bin'));row.final_parameters=modelPayload(directory+'/'+row.result.checkpoint+'/model.blzn');}
 save();console.log(JSON.stringify({kind,repeat,seconds:row.seconds,complete:row.complete}));
}
try{
 for(let i=0;i<repeats;++i)for(const kind of i%2?['compact','optimized','baseline']:['baseline','optimized','compact'])await run(kind,i);
 const control=report.rows.find(r=>r.kind==='baseline');report.fp32_payloads_identical=report.rows.filter(r=>r.kind!=='compact').every(r=>r.complete&&JSON.stringify(r.data_hashes)===JSON.stringify(control.data_hashes));
 report.fp32_parameters_identical=report.rows.filter(r=>r.kind!=='compact').every(r=>r.complete&&r.final_parameters===control.final_parameters);
 report.complete=report.rows.every(r=>r.complete)&&report.fp32_payloads_identical&&report.fp32_parameters_identical;
 if(!report.complete)process.exitCode=1;
}catch(error){report.error=String(error);process.exitCode=1;}
finally{report.finished=new Date().toISOString();save();}
