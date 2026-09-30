#!/usr/bin/env node
// Sequential, matched-work local comparison. Renderer and teacher ordering
// change intentionally; this is a throughput experiment, never a quality score.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn,execFileSync} from 'node:child_process';
import {hash} from './refactor-cycle.mjs';
const [directory,model,checkpoint,repeatsText='3']=process.argv.slice(2),repeats=Number(repeatsText);
if(!directory||!model||!checkpoint||!Number.isInteger(repeats)||repeats<1||repeats>5||fs.existsSync(directory))throw new Error('hardware-profile.mjs FRESH_DIRECTORY MODEL CHECKPOINT [REPEATS 1..5]');
const root=path.resolve(directory);fs.mkdirSync(root,{recursive:true});
const read=p=>JSON.parse(fs.readFileSync(p,'utf8')),sync=(p,args)=>execFileSync(p,args,{encoding:'utf8'}).trim();
const gpu=()=>sync('nvidia-smi',['--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu','--format=csv,noheader']);
const files=sync('git',['ls-files','-c','-o','--exclude-standard','-z']).split('\0').filter(p=>/\.(cpp|cu|hpp|cuh|vert|frag|mjs|nix)$|CMakeLists.txt$/.test(p)).sort();
const source=Object.fromEntries(files.map(p=>[p,hash(p)]));
const report={complete:false,score:null,started:new Date().toISOString(),revision:sync('git',['rev-parse','HEAD']),source_sha256:crypto.createHash('sha256').update(JSON.stringify(source)).digest('hex'),source,initial_model_sha256:hash(model),initial_checkpoint_sha256:hash(checkpoint),binary_sha256:hash('build/neural/blitz-neural-cycle'),gpu_before:gpu(),protocol_sha256:hash('research/PROTOCOL.md'),repeats,settings:{states:8,pool:4,conditions:6,preceding_states:6,updates:12288,batch:512,gpu_memory_mib:1024},rows:[]};
const save=()=>fs.writeFileSync(root+'/comparison.json',JSON.stringify(report,null,2)+'\n');save();
async function execute(kind,index){
 const run=root+'/'+kind+'-'+index,fd=fs.openSync(run+'.log','w');
 const args=kind==='native'?[run,'--states','8','--updates','2048','--minutes','3','--gpu-memory-mib','1024','--vertex-storage','fp32','--initialize',path.resolve(model),'--warmstart',path.resolve(checkpoint)]:['research/neural/pipeline-profile.mjs',run,path.resolve(model),path.resolve(checkpoint),'224768'];
 const binary=kind==='native'?'build/neural/blitz-neural-cycle':process.execPath,row={kind,index,gpu_before:gpu(),binary,args};report.rows.push(row);save();
 const start=performance.now(),child=spawn(binary,args,{stdio:['ignore',fd,fd]});let hard;
 const timer=setTimeout(()=>{child.kill('SIGTERM');hard=setTimeout(()=>child.kill('SIGKILL'),5000);},240000);
 try{row.code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',resolve);});}
 finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);row.seconds=(performance.now()-start)/1000;row.gpu_after=gpu();save();}
 const result=read(run+(kind==='native'?'/latest.json':'/report.json'));row.result=result;
 if(row.code!==0||!result.complete||result.states!==54||(kind==='native'?result.updates_this_invocation!==12288:result.final_step!==237056))throw new Error('Incomplete matched workload: '+kind+' '+index);
 save();console.log(JSON.stringify({kind,index,seconds:row.seconds}));
}
try{for(let i=0;i<repeats;++i)for(const kind of i%2?['native','legacy']:['legacy','native'])await execute(kind,i);report.complete=true;}
catch(error){report.error=String(error);process.exitCode=1;}
finally{report.finished=new Date().toISOString();save();}
