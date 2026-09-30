#!/usr/bin/env node
// Sequential A/B/B/A teacher comparison. Fixed work, inputs, labels and cameras;
// failures remain rows and never receive a speedup or quality score.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn,execFileSync} from 'node:child_process';
const [baseline,optimized,directory,model,only]=process.argv.slice(2);
if(!baseline||!optimized||!directory)throw new Error('teacher-profile.mjs BASELINE OPTIMIZED FRESH_OUTPUT [MODEL] [CASE]');
if(fs.existsSync(directory))throw new Error('Choose a fresh profile directory');fs.mkdirSync(directory,{recursive:true});
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const cases=[
 {name:'bench32',asset:'ph_painted_wooden_bench',pixels:32,previous:0,states:8},
 {name:'organic64',asset:'ph_sweet_potato',pixels:64,previous:0,states:8},
 {name:'bench-adjacent64',asset:'ph_painted_wooden_bench',pixels:64,previous:2,states:6},
 {name:'organic-adjacent64',asset:'ph_sweet_potato',pixels:64,previous:2,states:6},
 ...(model?[{name:'policy-adjacent64',asset:'ph_sweet_potato',pixels:64,previous:2,states:6,model},{name:'policy-adjacent32',asset:'ph_sweet_potato',pixels:32,previous:2,states:6,model}]:[])
].filter(c=>!only||c.name===only);
if(!cases.length)throw new Error('Unknown comparison case');
const report={baseline_sha256:hash(baseline),optimized_sha256:hash(optimized),model_sha256:model?hash(model):null,order:['baseline','optimized','optimized','baseline'],score:null,cases,rows:[]};
const save=()=>fs.writeFileSync(path.join(directory,'report.json'),JSON.stringify(report,null,2)+'\n');save();
const gpu=()=>execFileSync('nvidia-smi',['--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],{encoding:'utf8'}).trim();
for(const c of cases)for(const [repeat,variant] of report.order.entries()){
 const output=path.resolve(directory,`${c.name}-${repeat}-${variant}`),binary=path.resolve(variant==='baseline'?baseline:optimized);
 const args=[c.asset,output,'--states',String(c.states),'--pool','4','--pixels',String(c.pixels),'--previous-steps',String(c.previous),'--minutes','3','--gpu-memory-mib','256','--seed','101',...(c.model?['--model',path.resolve(c.model)]:[])];
 const before=gpu(),start=performance.now(),log=fs.openSync(output+'.log','w');
 const code=await new Promise(resolve=>{const child=spawn(binary,args,{stdio:['ignore',log,log]});child.on('error',error=>{fs.writeSync(log,String(error));resolve(-1);});child.on('exit',resolve);});fs.closeSync(log);
 const wall=(performance.now()-start)/1000,index=fs.existsSync(output+'/index.json')?JSON.parse(fs.readFileSync(output+'/index.json','utf8')):null;
 const row={case:c.name,variant,repeat,code,wall_seconds:wall,gpu_before:before,gpu_after:gpu(),args,index};
 if(fs.existsSync(output+'/reuse.json'))row.reuse=JSON.parse(fs.readFileSync(output+'/reuse.json','utf8'));
 report.rows.push(row);save();console.log(JSON.stringify({case:c.name,variant,repeat,code,wall_seconds:wall,teacher_seconds:index?.seconds,states:index?.states,sha256:index?.sha256}));
}
report.comparisons=cases.map(c=>{const rows=report.rows.filter(r=>r.case===c.name),complete=rows.every(r=>r.code===0&&r.index?.complete),hashes=[...new Set(rows.map(r=>r.index?.sha256))];
 const median=a=>{a.sort((x,y)=>x-y);return(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
 const metric=(variant,key)=>median(rows.filter(r=>r.variant===variant).map(r=>key==='seconds'?r.index.seconds:r.wall_seconds));
 return {case:c.name,complete,identical_training_payloads:complete&&hashes.length===1,hashes,...(complete?{teacher_speedup:metric('baseline','seconds')/metric('optimized','seconds'),wall_speedup:metric('baseline','wall')/metric('optimized','wall')}: {})};});
save();console.log(JSON.stringify(report.comparisons));
