// One offline comparison document; identical accessor bytes are stored once.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..'),run=promisify(execFile);
const read=async p=>JSON.parse(await fs.readFile(resolve(root,p),'utf8'));
const hash=b=>createHash('sha256').update(b).digest('hex');
const analysis=await read('research/chain-search/analysis.json'),pilot=await read('research/pilot.json');
const names={ph_painted_wooden_shelves:'Painted shelves',ph_metal_stool_02:'Metal stool',ph_grass_bermuda_01:'Bermuda grass',ph_dead_quiver_trunk:'Quiver trunk',ph_moon_rock_02:'Moon rock',ph_rock_face_02:'Rock face',si_3d_package_6c69a6bb_55e6_4356_8725_120ff7f8d652:'Bell X-1'};
const data={version:1,assets:pilot.assets.map(a=>({id:a.id,name:names[a.id]??a.title??a.id,category:a.category,source_url:a.source_url,license:a.license})),runs:[],payloads:{},comparisons:analysis.comparisons};
const insert=(g,bin,index)=>{
 const a=g.accessors[index],v=g.bufferViews[a.bufferView],width={SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[a.type],size={5121:1,5125:4,5126:4}[a.componentType];
 if(!width||!size||v.byteStride||a.sparse||v.buffer!==0)throw Error('Unsupported exported accessor');
 const start=(v.byteOffset??0)+(a.byteOffset??0),length=a.count*width*size;
 if(start<0||start+length>bin.length||length>v.byteLength)throw Error('Accessor outside export buffer');
 const bytes=bin.subarray(start,start+length),id=hash(bytes)+':'+a.componentType+':'+width;
 data.payloads[id]??={data:gzipSync(bytes).toString('base64'),type:a.componentType,width,count:a.count,bytes:length};return id;
};
for(const record of analysis.runs.filter(r=>r.split==='development')) {
 const inspect=`${record.dir}/inspection.json`;
 await run(resolve(root,'build/release/blitz-chain-inspect'),[record.dir,inspect],{cwd:root,maxBuffer:1024*1024});
 const inspections=await read(inspect);
 if(inspections.run_sha256!==record.metadata.run_sha256)throw Error('Inspection provenance mismatch');
 const item={id:record.id,preset:record.preset,method:record.method,score:record.score,complete:record.complete,config:record.metadata.config,
  binary_sha256:record.metadata.binary_sha256,run_sha256:record.metadata.run_sha256,dir:record.dir,assets:[]};
 for(const a of data.assets) {
  let row;try{row=await read(`${record.dir}/rows/${a.id}.json`);}catch(e){if(e.code!=='ENOENT')throw e;}
  if(!row||!row.complete||row.failed){item.assets.push({id:a.id,unavailable:true,failure:row?.failure??'Incomplete run'});continue;}
  const r=row.result,dir=`${record.dir}/meshes/${a.id}`;
  const [g,manifest,bin]=await Promise.all([read(dir+'/chain.gltf'),read(dir+'/lods.json'),fs.readFile(resolve(root,dir,'chain.bin'))]);
  if(bin.length!==r.storage.total_bytes||manifest.lods.length!==r.lods.length)throw Error('Export mismatch');
  const meshes=g.meshes.map(m=>({primitives:m.primitives.map(p=>({position:insert(g,bin,p.attributes.POSITION),normal:p.attributes.NORMAL==null?null:insert(g,bin,p.attributes.NORMAL),
   color:p.attributes.COLOR_0==null?null:insert(g,bin,p.attributes.COLOR_0),indices:insert(g,bin,p.indices),material:p.material,double_sided:!!g.materials[p.material].doubleSided}))}));
  let dense=null;try{dense=await read(`research/chain-search/audits/${record.id}/${a.id}.json`);}catch(e){if(e.code!=='ENOENT')throw e;}
  const digest=hash(bin),gltfHash=hash(await fs.readFile(resolve(root,dir,'chain.gltf')));
  if(dense&&(dense.run_sha256!==record.metadata.run_sha256||dense.chain_bin_sha256!==digest||dense.chain_gltf_sha256!==gltfHash||
     dense.row_sha256!==hash(await fs.readFile(resolve(root,record.dir,'rows',a.id+'.json')))))throw Error('Dense audit belongs to another export or measurement row');
  item.assets.push({id:a.id,result:r,meshes,nodes:g.nodes,inspection:inspections.assets[a.id],generation_seconds:row.generation_seconds,ratio:row.ratio,fallback:!!row.fallback,
   chain_sha256:digest,dir,dense,qualified:!row.fallback&&record.preset==='appearance'&&dense?.complete===true&&dense?.passed===true&&dense?.first_level===1});
 }
 data.runs.push(item);
}
const dir=resolve(root,'examples/reduction-board');await fs.mkdir(dir,{recursive:true});
const template=await fs.readFile(resolve(root,'tools/chain-search-board.html'),'utf8');
await fs.writeFile(dir+'/index.html',template.replace('__BOARD_DATA__',JSON.stringify(data).replaceAll('<','\\u003c')));
await fs.writeFile(dir+'/manifest.json',JSON.stringify({version:1,pilot_ids:data.assets.map(a=>a.id),runs:data.runs.map(r=>({id:r.id,run_sha256:r.run_sha256,binary_sha256:r.binary_sha256,config:r.config,assets:r.assets.map(a=>({id:a.id,unavailable:a.unavailable??false,chain_sha256:a.chain_sha256,qualified:a.qualified??false}))}))},null,2)+'\n');
console.log(JSON.stringify({runs:data.runs.length,assets:data.assets.length,unique_accessors:Object.keys(data.payloads).length,output:'examples/reduction-board/index.html'}));
