// Build a self-contained board from current-policy exported chains.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {dirname,resolve} from 'node:path';
import {entries} from './current-board-assets.mjs';

const root=resolve(import.meta.dirname,'..');
const read=async path=>JSON.parse(await fs.readFile(resolve(root,path),'utf8'));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const catalogs=await Promise.all([read('research/pilot.json'),read('research/corpus.json')]);
const assets=[];
for(const entry of entries){
  const [result,config,gltf,manifest,binary]=await Promise.all([
    read(entry.result),read(entry.config),read(entry.dir+'/chain.gltf'),read(entry.dir+'/lods.json'),
    fs.readFile(resolve(root,entry.dir,'chain.bin'))
  ]);
  const source=catalogs.flatMap(c=>c.assets).find(a=>a.id===entry.id);
  if(!source||source.split!=='development'||!source.opaque)throw Error('Missing development source provenance: '+entry.id);
  const inputBytes=await fs.readFile(resolve(root,entry.input));
  const inputGltf=JSON.parse(inputBytes);
  const inputBinary=await fs.readFile(resolve(root,dirname(entry.input),inputGltf.buffers[0].uri));
  const inputGltfSha256=digest(inputBytes),inputBinarySha256=digest(inputBinary);
  if(source.path===entry.input&&source.files?.some(f=>f.path===entry.input&&f.sha256!==inputGltfSha256))
    throw Error('Source glTF hash mismatch: '+entry.id);
  if(source.path===entry.input&&source.files?.some(f=>f.path.endsWith('/'+inputGltf.buffers[0].uri)&&f.sha256!==inputBinarySha256))
    throw Error('Source binary hash mismatch: '+entry.id);
  if(config.max_added_vertex_bytes_bps!==2000||config.triangle_overhead_bps!==0||
      result.max_added_vertex_bytes_bps!==2000||result.triangle_overhead_bps!==0||result.status!=='complete')
    throw Error('This is not a complete current-policy chain: '+entry.id);
  if(!(result.proposal_diagnostics?.tail_probe_evaluations>0)||
      result.proposal_diagnostics.tail_reserved_vertex_bytes>result.added_vertex_budget_bytes)
    throw Error('Missing capped tail-first search diagnostics: '+entry.id);
  if(result.proposal_diagnostics.adaptive_retry_selected&&!result.proposal_diagnostics.adaptive_retry_attempted||
      (result.proposal_diagnostics.adaptive_retry_evaluations??0)>result.candidate_evaluations)
    throw Error('Invalid adaptive retry diagnostics: '+entry.id);
  if(result.storage.total_bytes!==binary.length||result.storage.added_vertex_bytes>result.added_vertex_budget_bytes||
      manifest.storage.total_bytes!==binary.length)throw Error('Storage or budget mismatch: '+entry.id);
  if(result.runtime_levels.length!==result.runtime_storage.length||
      result.runtime_levels.some((index,i)=>index!==result.runtime_storage[i].scheduled_index)||
      result.runtime_storage.reduce((n,c)=>n+c.added_vertex_bytes,0)!==result.storage.added_vertex_bytes||
      result.runtime_storage.reduce((n,c)=>n+c.index_bytes,0)!==result.storage.index_bytes)
    throw Error('Runtime storage mismatch: '+entry.id);
  if(result.lods.some((l,i)=>l.triangles!==manifest.lods[i].triangles||
      (i&&(!l.adjacent.passed||!l.source.passed))))throw Error('Audit or export mismatch: '+entry.id);
  if(result.lods.at(-1)?.screen_pixels!==16)throw Error('Final scheduled size is not 16 px: '+entry.id);
  const firstPosition=gltf.meshes[gltf.nodes[0].mesh].primitives[0].attributes.POSITION;
  const bounds=gltf.accessors[firstPosition];
  const center=bounds.min.map((v,i)=>(v+bounds.max[i])/2);
  const radius=Math.hypot(...bounds.min.map((v,i)=>(bounds.max[i]-v)/2));
  assets.push({...entry,source_url:source.source_url,license:source.license,license_url:source.license_url,
    input_gltf_sha256:inputGltfSha256,input_binary_sha256:inputBinarySha256,
    chain_sha256:digest(binary),center,radius,
    binary:binary.toString('base64'),gltf:{accessors:gltf.accessors,bufferViews:gltf.bufferViews,
      meshes:gltf.meshes,nodes:gltf.nodes},
    lods:result.lods.map(l=>({triangles:l.triangles,vertices:l.vertices,shared_vertices:l.shared_vertices,
      screen_pixels:l.screen_pixels,transition_limit:l.transition_limit,adjacent_error:l.adjacent.error_px,
      source_error:l.source.error_px})),
    runtime_levels:result.runtime_levels,runtime_storage:result.runtime_storage,
    storage:result.storage,budget_bytes:result.added_vertex_budget_bytes,
    tail_probe_evaluations:result.proposal_diagnostics.tail_probe_evaluations,
    tail_reserved_vertex_bytes:result.proposal_diagnostics.tail_reserved_vertex_bytes,
    adaptive_retry_attempted:!!result.proposal_diagnostics.adaptive_retry_attempted,
    adaptive_retry_selected:!!result.proposal_diagnostics.adaptive_retry_selected,
    adaptive_retry_evaluations:result.proposal_diagnostics.adaptive_retry_evaluations??0,
    generation_seconds:result.generation_seconds,candidate_evaluations:result.candidate_evaluations,
    candidate_budget:config.candidate_budget,levels:config.levels,config:entry.config});
}
const data={version:1,policy:{max_added_vertex_bytes_bps:2000,triangle_overhead_bps:0},assets};
const template=await fs.readFile(resolve(root,'tools/current-board.html'),'utf8');
const output=template.replace('__BOARD_DATA__',JSON.stringify(data).replaceAll('<','\\u003c'))
  .replaceAll('__EXAMPLE_COUNT__',String(assets.length));
const directory=resolve(root,'examples/current-board');
await fs.mkdir(directory,{recursive:true});
await fs.writeFile(resolve(directory,'index.html'),output);
await fs.writeFile(resolve(directory,'manifest.json'),JSON.stringify({version:1,policy:data.policy,
  assets:assets.map(({id,name,dir,input,input_gltf_sha256,input_binary_sha256,source_url,license,chain_sha256,config,runtime_levels,budget_bytes,storage,tail_probe_evaluations,tail_reserved_vertex_bytes,adaptive_retry_attempted,adaptive_retry_selected,adaptive_retry_evaluations})=>
    ({id,name,dir,input,input_gltf_sha256,input_binary_sha256,source_url,license,chain_sha256,config,runtime_levels,budget_bytes,storage,tail_probe_evaluations,tail_reserved_vertex_bytes,adaptive_retry_attempted,adaptive_retry_selected,adaptive_retry_evaluations}))},null,2)+'\n');
console.log(JSON.stringify({assets:assets.length,bytes:Buffer.byteLength(output),output:'examples/current-board/index.html'}));
