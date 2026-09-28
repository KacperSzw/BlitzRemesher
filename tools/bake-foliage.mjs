// Unscored, geometry-only chain demonstrations. The C++ library does every
// reduction/audit; this driver checkpoints inputs, timings and exported hashes.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {spawn,execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {cpus} from 'node:os';
import {resolve,dirname} from 'node:path';
const plan=JSON.parse(await readFile('research/foliage/chains.json','utf8'));
const input=resolve(process.argv[2]||'build/foliage/geometry'),binary=resolve(process.argv[3]||'build/foliage-release/blitz');
const only=new Set(process.argv.slice(4)),run='research/runs/'+plan.run,hash=x=>createHash('sha256').update(x).digest('hex');
const source=JSON.parse(await readFile(input+'/manifest.json','utf8'));
if(source.scope!=='foliage_card_geometry_only'||source.benchmark_eligible!==false)throw Error('Explicit geometry-only inputs required');
const cache=await readFile(dirname(binary)+'/CMakeCache.txt','utf8'),compiler=cache.match(/^CMAKE_CXX_COMPILER:FILEPATH=(.+)$/m)?.[1];if(!compiler)throw Error('Missing compiler provenance');
const sources=execFileSync('git',['ls-files','-z','--','include','src','tools/main.cpp','tools/benchmark.cpp','CMakeLists.txt'],{encoding:'utf8'}).split('\0').filter(Boolean).sort();const sourceHash=createHash('sha256');for(const p of sources){sourceHash.update(p+'\0');sourceHash.update(await readFile(p));}
const metadata={version:1,scope:source.scope,scored:false,config:plan.config,jobs:plan.examples,source_manifest_sha256:source.manifest_sha256,geometry_manifest_sha256:hash(await readFile(input+'/manifest.json')),binary_sha256:hash(await readFile(binary)),protocol_sha256:hash(await readFile('research/PROTOCOL.md')),source_tree_sha256:sourceHash.digest('hex'),git_revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),cpu:cpus()[0].model,compiler:execFileSync(compiler,['--version'],{encoding:'utf8'}).split('\n')[0],workers:2,timing_scope:'generation_seconds excludes import and export; includes all search and audit work',vertex_contract:'Reuse shares immutable imported LOD0 positions and UV0; node transforms were applied during geometry extraction'};
metadata.run_sha256=hash(JSON.stringify(metadata));
await mkdir(run+'/rows',{recursive:true});await mkdir(run+'/configs',{recursive:true});
try{const previous=JSON.parse(await readFile(run+'/metadata.json','utf8'));if(previous.run_sha256!==metadata.run_sha256)throw Error('Frozen run inputs changed; use a new run directory');}catch(e){if(e.code!=='ENOENT')throw e;}
await writeFile(run+'/metadata.json',JSON.stringify(metadata,null,2)+'\n');
const start=Date.now(),jobs=plan.examples.filter(e=>!only.size||only.has(e.id)),failures=[];
async function bake(job){
  const path=run+'/rows/'+job.id+'.json';
  try{const prior=JSON.parse(await readFile(path,'utf8'));if(prior.complete&&prior.run_sha256===metadata.run_sha256){if(hash(await readFile(run+'/meshes/'+job.id+'/chain.bin'))!==prior.output_sha256||hash(await readFile(run+'/meshes/'+job.id+'/chain.gltf'))!==prior.gltf_sha256)throw Error('Changed recorded output');console.log('Resume verified row '+job.id);return;}}catch(e){if(e.code!=='ENOENT')throw e;}
  const record=source.assets.find(a=>a.id===job.id);if(!record)throw Error('Missing source '+job.id);
  const model=input+'/'+job.id+'/source.gltf',bytes=input+'/'+job.id+'/source.bin';if(hash(await readFile(model))!==record.gltf_sha256||hash(await readFile(bytes))!==record.binary_sha256)throw Error('Changed geometry input');
  const config={...plan.config,output:job.output};await writeFile(run+'/configs/'+job.id+'.json',JSON.stringify(config,null,2)+'\n');
  console.log('Bake '+job.id+' / '+job.output);const wall=performance.now();
  const result=await new Promise((resolve,reject)=>{
    const child=spawn(binary,['simplify',model,'--config',run+'/configs/'+job.id+'.json','--out',run+'/meshes/'+job.id],{stdio:['ignore','pipe','pipe']});let stdout='',stderr='';
    const timer=setTimeout(()=>child.kill('SIGTERM'),Math.max(1,50*60*1000-(Date.now()-start)));child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);child.on('error',e=>{clearTimeout(timer);reject(e);});child.on('close',code=>{clearTimeout(timer);resolve({code,stdout,stderr});});
  });
  const row={id:job.id,run_sha256:metadata.run_sha256,scope:source.scope,scored:false,complete:false,failed:result.code!==0,seconds:(performance.now()-wall)/1000,config,output_mode:job.output,input:record,stderr:result.stderr};
  try{row.result=JSON.parse(result.stdout);if(row.result.status!=='complete'||result.code!==0)throw Error('Incomplete bake');row.generation_seconds=row.result.generation_seconds;row.export_seconds=row.result.export_seconds;
    if(!Number.isFinite(row.generation_seconds)||row.generation_seconds<=0||row.result.lods.length!==config.levels)throw Error('Missing bake measurement');
    if(row.result.lods[0].triangles!==record.triangles)throw Error('Imported source triangle count changed');
    for(const l of row.result.lods)if(!l.source.passed||!l.adjacent.passed)throw Error('Failed geometry gate');
    if(job.output==='reuse'&&!row.result.lods.every(l=>l.shared_vertices))throw Error('Reuse emitted new vertices');
    row.output_sha256=hash(await readFile(run+'/meshes/'+job.id+'/chain.bin'));row.gltf_sha256=hash(await readFile(run+'/meshes/'+job.id+'/chain.gltf'));row.ratio=row.result.lods.slice(1).reduce((n,l)=>n+l.triangles,0)/(record.triangles*(config.levels-1));row.complete=true;
  }catch(e){row.failed=true;row.failure=e.message;row.raw_stdout=result.stdout;failures.push(job.id);}
  await writeFile(path,JSON.stringify(row,null,2)+'\n');console.log((row.complete?'Complete ':'Failed ')+job.id+' '+(row.generation_seconds?.toFixed(2)??'?')+' s');
}
let next=0;await Promise.all([0,1].map(async()=>{while(next<jobs.length&&Date.now()-start<50*60*1000)await bake(jobs[next++]);}));
const rows=[];for(const e of plan.examples){try{rows.push(JSON.parse(await readFile(run+'/rows/'+e.id+'.json','utf8')));}catch(e){if(e.code!=='ENOENT')throw e;}}
await writeFile(run+'/summary.json',JSON.stringify({run_sha256:metadata.run_sha256,scope:source.scope,scored:false,complete:rows.length===plan.examples.length&&rows.every(r=>r.complete),expected:plan.examples.length,completed:rows.filter(r=>r.complete).length,failures,score:null,reason:'Opaque card geometry demonstrations; opacity/texture/shading are not audited'},null,2)+'\n');
if(failures.length)process.exitCode=1;
