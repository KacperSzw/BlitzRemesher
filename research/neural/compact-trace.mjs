#!/usr/bin/env node
// Nsight Systems 2026 SQLite export. GPU active time and host API time are
// separate views of overlapping work; never add them to cycle wall time.
import fs from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {hash} from './refactor-cycle.mjs';
const [database,cycle,output]=process.argv.slice(2);
if(!database||!cycle||!output||fs.existsSync(output))throw Error('compact-trace.mjs TRACE.sqlite CYCLE_REPORT.json FRESH_REPORT.json');
const db=new DatabaseSync(database,{readOnly:true}),report=JSON.parse(fs.readFileSync(cycle,'utf8'));
const groups=(table,key='nameId',where='1')=>db.prepare(`SELECT s.value AS name,count(*) AS calls,sum(t.end-t.start)*1e-9 AS seconds FROM ${table} t JOIN StringIds s ON s.id=t.${key} WHERE ${where} GROUP BY s.value ORDER BY seconds DESC`).all();
const total=rows=>({calls:rows.reduce((n,x)=>n+x.calls,0),seconds:rows.reduce((n,x)=>n+x.seconds,0)});
const graph=groups('CUPTI_ACTIVITY_KIND_KERNEL','demangledName','t.graphId<>0');
const other=groups('CUPTI_ACTIVITY_KIND_KERNEL','demangledName','t.graphId=0 OR t.graphId IS NULL');
const phases=Object.groupBy(report.phases,p=>p.phase),phaseSeconds=Object.fromEntries(Object.entries(phases).map(([k,v])=>[k,v.reduce((n,p)=>n+p.seconds,0)]));
const training=phases.training??[],trainingParts=Object.fromEntries(['append_seconds','capture_seconds','update_seconds','checkpoint_seconds'].map(k=>[k,training.reduce((n,p)=>n+(p[k]??0),0)]));
const result={score:null,cycle_complete:report.complete,database_sha256:hash(database),cycle_sha256:hash(cycle),
  note:'Instrumented single run on a shared GPU. GPU durations are sums of activity intervals. Host API calls include GPU waits; background checkpoint times overlap. These are not additive wall-time categories.',
  cycle_seconds:report.seconds,phase_seconds:phaseSeconds,unattributed_seconds:report.seconds-Object.values(phaseSeconds).reduce((a,b)=>a+b,0),training_parts:trainingParts,checkpoint:report.checkpoint_timing,
  training_graph:{...total(graph),updates:report.updates_this_invocation,kernels_per_update:total(graph).calls/report.updates_this_invocation,kernels:graph},
  other_cuda:{...total(other),kernels:other},cuda_api:groups('CUPTI_ACTIVITY_KIND_RUNTIME'),vulkan_api:groups('VULKAN_API'),vulkan_gpu:groups('VULKAN_WORKLOAD'),
  copies:db.prepare('SELECT copyKind AS kind,count(*) AS calls,sum(bytes) AS bytes,sum(end-start)*1e-9 AS seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').all()};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({cycle_seconds:result.cycle_seconds,phase_seconds:phaseSeconds,training_graph:total(graph),other_cuda:total(other),vulkan_gpu:total(result.vulkan_gpu)}));
