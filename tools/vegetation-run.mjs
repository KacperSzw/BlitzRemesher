// Checkpointed geometry-only experiments. The C++ executable owns reduction and audits.
import fs from 'node:fs/promises';
import {spawn,execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {cpus} from 'node:os';
import {resolve,dirname} from 'node:path';
const [planPath,binaryArg='build/vegetation-release/blitz',geometryArg='build/foliage/geometry',...filters]=process.argv.slice(2);
if(!planPath)throw Error('vegetation-run.mjs PLAN [BINARY GEOMETRY [RUN_SUFFIX...]]');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),sha=x=>createHash('sha256').update(x).digest('hex');
const plan=await read(planPath),binary=resolve(binaryArg),geometry=resolve(geometryArg),inputs=await read(geometry+'/manifest.json');
const collection=await read('research/foliage/manifest.json');
if(inputs.scope!=='foliage_card_geometry_only'||inputs.benchmark_eligible!==false)throw Error('Explicit geometry-only inputs required');
const cache=await fs.readFile(dirname(binary)+'/CMakeCache.txt','utf8'),compiler=cache.match(/^CMAKE_CXX_COMPILER:FILEPATH=(.+)$/m)?.[1];
const names=execFileSync('git',['ls-files','--cached','--others','--exclude-standard','-z','--','src','include','tools/main.cpp','tools/benchmark.cpp','tools/vegetation_meshopt.cpp','CMakeLists.txt','cmake'],{encoding:'utf8'}).split('\0').filter(Boolean).sort();
const tree=createHash('sha256');for(const p of [...new Set(names)]){tree.update(p+'\0');tree.update(await fs.readFile(p));}
const identity={binary_sha256:sha(await fs.readFile(binary)),source_tree_sha256:tree.digest('hex'),cmake_cache_sha256:sha(cache),runner_sha256:sha(await fs.readFile(new URL(import.meta.url))),git_revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),protocol_sha256:sha(await fs.readFile('research/PROTOCOL.md')),geometry_manifest_sha256:sha(await fs.readFile(geometry+'/manifest.json')),source_manifest_sha256:inputs.manifest_sha256,cpu:cpus()[0].model,compiler:execFileSync(compiler,['--version'],{encoding:'utf8'}).split('\n')[0],threads_per_bake:1};
const workers=plan.workers??2;if(!Number.isInteger(workers)||workers<1||workers>4)throw Error('Worker limit is 1..4');
const deadline=Date.now()+50*60*1000,jobs=[],groups=[];
for(const variant of plan.variants)for(const output of plan.modes)for(const budget of plan.budgets){
 const suffix=`${variant.name}-${output}-b${budget}`;if(filters.length&&!filters.includes(suffix))continue;
 const run=`research/runs/${plan.name}-${suffix}`,config={...plan.config,output,candidate_budget:budget,research:{...variant.research,trace:plan.trace??true}};
 const metadata={version:1,scope:inputs.scope,scored:false,config,variant:variant.name,extra_args:variant.args??[],cohort:plan.assets,workers,plan_sha256:sha(await fs.readFile(planPath)),...identity,timing_scope:'Generation/search/audit; import and export measured separately; shared workstation'};
 metadata.run_sha256=sha(JSON.stringify(metadata));await fs.mkdir(run+'/rows',{recursive:true});await fs.mkdir(run+'/configs',{recursive:true});await fs.mkdir(run+'/.scratch',{recursive:true});
 try{const old=await read(run+'/metadata.json');if(old.run_sha256!==metadata.run_sha256)throw Error('Run identity changed: '+run);}catch(e){if(e.code!=='ENOENT')throw e;}
 await fs.writeFile(run+'/metadata.json',JSON.stringify(metadata,null,2)+'\n');groups.push({run,metadata});for(const id of plan.assets)jobs.push({id,run,config,metadata});
}
async function bake({id,run,config,metadata}){
 const path=run+'/rows/'+id+'.json';try{const old=await read(path);if(old.complete&&old.run_sha256===metadata.run_sha256){
  if(sha(await fs.readFile(run+'/meshes/'+id+'/chain.bin'))!==old.output_sha256||sha(await fs.readFile(run+'/meshes/'+id+'/chain.gltf'))!==old.gltf_sha256)throw Error('Output changed: '+id);return;
 }}catch(e){if(e.code!=='ENOENT')throw e;}
 const record=inputs.assets.find(x=>x.id===id);if(!record)throw Error('Missing input '+id);
 const model=geometry+'/'+id+'/source.gltf';if(sha(await fs.readFile(model))!==record.gltf_sha256||sha(await fs.readFile(geometry+'/'+id+'/source.bin'))!==record.binary_sha256)throw Error('Input hash changed');
 const cfg=run+'/configs/'+id+'.json';await fs.writeFile(cfg,JSON.stringify(config,null,2)+'\n');const begin=performance.now(),rss=run+'/.scratch/'+id+'.rss';
 const r=await new Promise((done,fail)=>{const child=spawn('/run/current-system/sw/bin/time',['-f','%M','-o',rss,binary,'simplify',model,'--config',cfg,'--out',run+'/meshes/'+id,...metadata.extra_args],{detached:true,stdio:['ignore','pipe','pipe']});let stdout='',stderr='';
  const timer=setTimeout(()=>{try{process.kill(-child.pid,'SIGTERM');}catch{}},Math.max(1,deadline-Date.now()));child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);child.on('error',e=>{clearTimeout(timer);fail(e);});child.on('close',code=>{clearTimeout(timer);done({code,stdout,stderr});});});
 const row={id,category:collection.assets.find(a=>a.id===id).category,run_sha256:metadata.run_sha256,scope:inputs.scope,scored:false,complete:false,failed:r.code!==0,config,output_mode:config.output,input:record,seconds:(performance.now()-begin)/1000,stderr:r.stderr};
 try{row.result=JSON.parse(r.stdout);if(r.code!==0||row.result.status!=='complete')throw Error('Incomplete bake');const levels=row.result.lods;
  if(levels.length!==config.levels||levels[0].triangles!==record.triangles||levels.some((l,i)=>!l.source.passed||!l.adjacent.passed||(i&&l.triangles>levels[i-1].triangles)))throw Error('Invalid chain');
  if(config.output==='reuse'&&levels.some(l=>!l.shared_vertices))throw Error('Reuse contract failed');
  row.generation_seconds=row.result.generation_seconds;row.export_seconds=row.result.export_seconds;row.peak_rss_kib=Number((await fs.readFile(rss,'utf8')).trim());
  row.output_sha256=sha(await fs.readFile(run+'/meshes/'+id+'/chain.bin'));row.gltf_sha256=sha(await fs.readFile(run+'/meshes/'+id+'/chain.gltf'));
  row.ratio=levels.slice(1).reduce((n,l)=>n+l.triangles,0)/(record.triangles*(levels.length-1));row.final_ratio=levels.at(-1).triangles/record.triangles;row.tail_ratio=levels.slice(-3).reduce((n,l)=>n+l.triangles,0)/(3*record.triangles);row.complete=true;
 }catch(e){row.failed=true;row.failure=e.message;row.raw_stdout=r.stdout;}
 await fs.writeFile(path,JSON.stringify(row,null,2)+'\n');console.log(JSON.stringify({run:run.split('/').at(-1),id,complete:row.complete,triangles:row.result?.lods.map(l=>l.triangles),seconds:row.generation_seconds}));
}
let next=0;await Promise.all(Array.from({length:workers},async()=>{while(next<jobs.length&&Date.now()<deadline)await bake(jobs[next++]);}));
let incomplete=false;
for(const {run,metadata} of groups){const rows=[];for(const id of plan.assets)try{rows.push(await read(run+'/rows/'+id+'.json'));}catch(e){if(e.code!=='ENOENT')throw e;}
 const complete=rows.length===plan.assets.length&&rows.every(r=>r.complete&&!r.failed&&r.run_sha256===metadata.run_sha256);incomplete||=!complete;
 await fs.writeFile(run+'/summary.json',JSON.stringify({run_sha256:metadata.run_sha256,scope:inputs.scope,scored:false,score:null,expected:plan.assets.length,completed:rows.filter(r=>r.complete).length,complete,failures:rows.filter(r=>r.failed).map(r=>r.id)},null,2)+'\n');}
if(incomplete)process.exitCode=2;
