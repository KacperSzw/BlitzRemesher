import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const plan=JSON.parse(await readFile('research/foliage/chains.json','utf8')),root=process.argv[2]||'research/runs/'+plan.run;
const metadata=JSON.parse(await readFile(root+'/metadata.json','utf8')),hash=b=>createHash('sha256').update(b).digest('hex');
const jobs=process.argv[2]?metadata.cohort.map(id=>({id,output:metadata.config.output})):plan.examples;
const report={run_sha256:metadata.run_sha256,scope:'card_geometry_only',scored:false,complete:false,assets:[],geometry_gates:0};
for(const job of jobs){
  const row=JSON.parse(await readFile(root+'/rows/'+job.id+'.json','utf8'));
  const text=await readFile(root+'/meshes/'+job.id+'/chain.gltf'),g=JSON.parse(text),binary=await readFile(root+'/meshes/'+job.id+'/chain.bin');
  assert.equal(row.run_sha256,metadata.run_sha256);assert.equal(hash(text),row.gltf_sha256);assert.equal(hash(binary),row.output_sha256);assert.ok(row.complete&&!row.failed);
  assert.ok(row.generation_seconds>0&&row.export_seconds>=0&&row.seconds>=row.generation_seconds+row.export_seconds);
  let previous=Infinity,sourceAttributes=null;const triangles=[],shared=[];
  const indexData=a=>{const v=g.bufferViews[a.bufferView];assert.equal(a.componentType,5125);assert.equal(a.type,'SCALAR');assert.equal(v.buffer,0);const start=(v.byteOffset||0)+(a.byteOffset||0);assert.ok(start+a.count*4<=binary.length);return binary.subarray(start,start+a.count*4);};
  for(const [i,lod] of row.result.lods.entries()){
    assert.ok(lod.triangles<=previous);previous=lod.triangles;
    for(const [kind,limit] of [['source',lod.source_limit],['adjacent',lod.transition_limit]]){assert.ok(lod[kind].passed&&lod[kind].complete);assert.ok(lod[kind].error_px<=limit+1e-12);report.geometry_gates++;}
    const mesh=g.meshes[g.nodes[i].mesh];let count=0;
    for(const p of mesh.primitives){const ix=g.accessors[p.indices],vertices=g.accessors[p.attributes.POSITION].count;const bytes=indexData(ix);count+=ix.count/3;
      for(let j=0;j<ix.count;j++)assert.ok(bytes.readUInt32LE(j*4)<vertices);
      if(i===0)sourceAttributes=p.attributes;
      if(job.output==='reuse'){assert.ok(lod.shared_vertices);assert.deepEqual(p.attributes,sourceAttributes,'Every reuse level must bind the identical LOD0 vertex accessors');}
    }
    assert.equal(count,lod.triangles);triangles.push(count);shared.push(lod.shared_vertices);
  }
  assert.equal(triangles[0],row.input.triangles);assert.equal(triangles.length,metadata.config.levels);
  report.assets.push({id:job.id,triangles,shared_vertices:shared,bake_seconds:row.generation_seconds,output:job.output,runtime_levels:row.result.runtime_lod_count,passed:true});
}
report.complete=true;await writeFile(process.argv[3]||'research/foliage/chain-checks.json',JSON.stringify(report,null,2)+'\n');console.log(report.assets.length+' chains, '+report.geometry_gates+' geometry gates, exported indices and shared vertex accessors verified.');
