// Package recorded neural meshes into a portable, offline inspection page.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const output=path.resolve(process.argv[2]??path.join(root,'research/neural/viewer'));
const evidence='research/neural/evidence/action-screening-v3/final/pilot-2';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p)));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const geometries={},files=[];
function accessor(g,bin,id,component,type,width){
  const a=g.accessors[id],v=g.bufferViews[a?.bufferView];
  assert.ok(a&&v&&!a.sparse&&a.componentType===component&&a.type===type&&v.buffer===0,'Unsupported exported accessor');
  const start=v.byteOffset??0,offset=a.byteOffset??0,stride=v.byteStride??width;
  assert.ok(a.count>0&&stride>=width&&offset>=0&&start>=0&&start+v.byteLength<=bin.length);
  assert.ok(offset+(a.count-1)*stride+width<=v.byteLength,'Accessor exceeds recorded buffer');
  const out=Buffer.alloc(a.count*width);
  for(let i=0;i<a.count;i++)bin.copy(out,i*width,start+offset+i*stride,start+offset+i*stride+width);
  return out;
}
function chain(method,id){
  const base=evidence+'/'+method+'/meshes/'+id;
  for(const name of ['chain.gltf','chain.bin','lods.json']){const bytes=fs.readFileSync(path.join(root,base,name));files.push({path:base+'/'+name,sha256:sha(bytes)});}
  const g=read(base+'/chain.gltf'),meta=read(base+'/lods.json'),bin=fs.readFileSync(path.join(root,base,'chain.bin'));
  assert.equal(g.asset.generator,'BlitzRemesher');assert.equal(g.buffers.length,1);assert.equal(g.buffers[0].uri,'chain.bin');
  assert.equal(g.buffers[0].byteLength,bin.length);assert.equal(meta.status,'complete');
  const levels=meta.lods.map((lod,level)=>{
    const node=g.nodes[lod.gltf_node];assert.equal(node.mesh,lod.gltf_mesh);
    assert.ok(!node.matrix&&!node.translation&&!node.rotation&&!node.scale,'Exported coordinates must be untransformed');
    const primitives=g.meshes[node.mesh].primitives;assert.equal(primitives.length,1);
    const p=primitives[0];assert.equal(p.mode??4,4);
    const positions=accessor(g,bin,p.attributes.POSITION,5126,'VEC3',12),normals=accessor(g,bin,p.attributes.NORMAL,5126,'VEC3',12),uv=accessor(g,bin,p.attributes.TEXCOORD_0,5126,'VEC2',8),indices=accessor(g,bin,p.indices,5125,'SCALAR',4);
    assert.equal(normals.length,positions.length);assert.equal(uv.length/8,positions.length/12);assert.equal(indices.length/12,lod.triangles);
    for(const stream of [positions,normals,uv])for(let i=0;i<stream.length;i+=4)assert.ok(Number.isFinite(stream.readFloatLE(i)));
    for(let i=0;i<indices.length;i+=4)assert.ok(indices.readUInt32LE(i)<positions.length/12);
    if(level)assert.ok(lod.source.complete&&lod.source.passed&&lod.adjacent.complete&&lod.adjacent.passed);
    const key=sha(Buffer.concat([positions,normals,uv,indices]));
    geometries[key]??=Object.fromEntries(Object.entries({positions,normals,uv,indices}).map(([k,v])=>[k,v.toString('base64')]));
    return {geometry:key,triangles:lod.triangles,pixels:lod.screen_pixels,source_error:lod.source.error_px,source_limit:lod.source_limit,adjacent_error:lod.adjacent.error_px,adjacent_limit:lod.transition_limit};
  });
  return {levels,chain_ratio:read(evidence+'/'+method+'/rows/'+id+'.json').ratio};
}
const assets=[['ph_moon_rock_02','Moon rock 02','moon_rock_02',.55,.28],['ph_painted_wooden_shelves','Painted shelves','painted_wooden_shelves',-.68,.14]].map(([id,name,slug,yaw,pitch])=>{
  const constant=chain('constant',id),learned=chain('learned-0',id);
  assert.equal(constant.levels.length,learned.levels.length);assert.equal(constant.levels[0].geometry,learned.levels[0].geometry,'Source differs between methods');
  const positions=Buffer.from(geometries[learned.levels[0].geometry].positions,'base64'),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<positions.length;i+=12)for(let k=0;k<3;k++){const x=positions.readFloatLE(i+4*k);lo[k]=Math.min(lo[k],x);hi[k]=Math.max(hi[k],x);}
  const center=lo.map((x,k)=>(x+hi[k])/2),radius=Math.hypot(...hi.map((x,k)=>x-lo[k]))/2;
  assert.ok(radius>0);return {id,name,source_url:'https://polyhaven.com/a/'+slug,center,radius,yaw,pitch,constant,learned};
});
const data={revision:'46ba983',seed:101,updates:16384,assets,geometries};
const template=fs.readFileSync(path.join(root,'tools/neural-viewer.html'),'utf8');assert.equal(template.split('@NEURAL_DATA@').length,2);
fs.mkdirSync(output,{recursive:true});fs.writeFileSync(output+'/index.html',template.replace('@NEURAL_DATA@',JSON.stringify(data).replaceAll('<','\\u003c')));
fs.writeFileSync(output+'/manifest.json',JSON.stringify({revision:data.revision,seed:data.seed,updates:data.updates,files,assets:assets.map(a=>({id:a.id,source:a.learned.levels[0].triangles,constant:a.constant.levels.map(l=>l.triangles),learned:a.learned.levels.map(l=>l.triangles)}))},null,2)+'\n');
console.log(`Packaged ${assets.length} assets, ${Object.keys(geometries).length} exact saved geometries: ${output}/index.html`);
