import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {Api,Rental,GPU,IMAGE,chooseQuote,podRequest,verifyPod,terminationDue,rentalDeadlines,retrySsh} from '../research/neural/runpod-api.mjs';
import {deployment,verifyDevice} from '../research/neural/runpod-profile.mjs';
const gpu={id:GPU,memory:96,secure:true,price:{secure:2.09},dataCenters:[{id:'EU-1',availability:'HIGH'}]};
const centers=[{id:'EU-1',networkVolumeTypes:['STANDARD']}];
const initial=()=>({name:'test-unique',quote:chooseQuote([gpu],centers),setup_deadline_ms:1800000,deadline_ms:7200000});
test('repeatable SSH calls recover transport failures, preserve command failures and stop retrying',async()=>{
  let calls=0,waits=0;
  const network=Object.assign(new Error('connection timeout'),{ssh_exit:255});
  const result=await retrySsh(async()=>{if(++calls<3)throw network;return 'ready';},async()=>{++waits;});
  assert.equal(result,'ready');assert.equal(calls,3);assert.equal(waits,2);
  for(const code of [1,2]){
    calls=0;const command=Object.assign(new Error('command failed'),{ssh_exit:code});
    await assert.rejects(retrySsh(async()=>{++calls;throw command;},async()=>{}),/command failed/);
    assert.equal(calls,1);
  }
  calls=0;await assert.rejects(retrySsh(async()=>{++calls;throw network;},async()=>{},3),/connection timeout/);
  assert.equal(calls,3);
});
test('recovery can shorten a rental to its original cutoff without extending the budget',()=>{
  const started=12340000,originalEnd=started+90*60000;
  const limits=rentalDeadlines(started,originalEnd);
  assert.equal(limits.deadline_ms,originalEnd);
  assert.equal(limits.training_deadline_ms,originalEnd-10*60000);
  assert.ok(limits.setup_deadline_ms<=started+30*60000);
  for(const minutes of [15,45,160]){
    const d=rentalDeadlines(started,started+minutes*60000);
    assert.ok(d.setup_deadline_ms<=d.training_deadline_ms);
    assert.ok(d.training_deadline_ms<d.deadline_ms);
  }
  assert.equal(rentalDeadlines(started).training_minutes,120);
  for(const deadline of [NaN,Infinity,started,started+10*60000,started+161*60000])
    assert.throws(()=>rentalDeadlines(started,deadline));
});
test('status succeeds with optional state files absent and reports every available record',()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'blitz-cloud-status-'));
  const cli=fileURLToPath(new URL('../research/neural/runpod.mjs',import.meta.url));
  try{
    fs.writeFileSync(directory+'/rental.json',JSON.stringify({name:'fixture-rental'}));
    for(const collected of [false,true]){
      if(collected)fs.writeFileSync(directory+'/collection.json',JSON.stringify({verified:true}));
      const result=spawnSync(process.execPath,[cli,'status',directory],{encoding:'utf8'});
      assert.equal(result.status,0,result.stderr);assert.match(result.stdout,/fixture-rental/);
      assert.equal(result.stdout.includes('collection.json'),collected);
    }
  }finally{fs.rmSync(directory,{recursive:true,force:true});}
});
class MockApi {
  constructor(){this.calls=[];this.volumes=[];this.resources=[];this.lost=false;this.deletionFails=false;}
  async pods(){return this.resources.map(p=>({...p}));}
  async request(method,uri,body){
    this.calls.push({method,uri,body});
    if(uri==='/network-volumes'){
      if(method==='GET')return {networkVolumes:this.volumes};
      const v={...body,id:'volume1'};this.volumes.push(v);return v;
    }
    if(uri==='/pods'&&method==='POST'){
      const pod={...body,id:'pod1'};this.resources.push(pod);
      if(this.lost)throw new Error('connection lost after create');return pod;
    }
    if(uri.startsWith('/pods/')){
      const id=decodeURIComponent(uri.slice(6));
      if(method==='DELETE'){if(this.deletionFails)throw new Error('temporary API failure');this.resources=this.resources.filter(p=>p.id!==id);return null;}
      return this.resources.find(p=>p.id===id)??null;
    }
    if(uri.startsWith('/network-volumes/')&&method==='DELETE'){this.volumes=[];return null;}
    throw new Error('Unexpected mock request');
  }
}
test('reject wrong GPUs, unknown prices, insufficient VRAM and unavailable storage',()=>{
  for(const invalid of [{...gpu,id:'NVIDIA GeForce RTX 5090'},{...gpu,id:GPU+' MIG 2g.48gb'},
    {...gpu,memory:48},{...gpu,secure:false},
    {...gpu,price:{secure:2.51}},{...gpu,price:{secure:NaN}},{...gpu,dataCenters:[{id:'EU-1',availability:'NONE'}]}])
    assert.throws(()=>chooseQuote([invalid],centers));
  assert.throws(()=>chooseQuote([gpu],[]));
  assert.throws(()=>chooseQuote([gpu],centers,'unavailable-location'));
  assert.equal(chooseQuote([gpu],centers,'EU-1').data_center,'EU-1');
  for(const rate of [1.9,2.09,2.5])assert.equal(chooseQuote([{...gpu,price:{secure:rate}}],centers).gpu_hourly_usd,rate);
});
test('Pod request is pinned and forwards only the public key',()=>{
  const s={...initial(),volume_id:'volume1'},body=podRequest(s,'ssh-ed25519 public-test\n');
  assert.equal(body.image,IMAGE);assert.match(body.image,/@sha256:[a-f0-9]{64}$/);
  assert.equal(body.gpu.count,1);assert.equal(body.cloud,'SECURE');
  assert.deepEqual(body.env,{PUBLIC_KEY:'ssh-ed25519 public-test'});
  assert.deepEqual(body.ports,['22/tcp']);assert.equal(body.mounts.network[0].volumeId,'volume1');
});
test('actual allocation must satisfy hardware and rate caps',()=>{
  const pod={cloud:'SECURE',cost:2.10,gpu:{id:GPU,count:1,memory:deployment.host_ram_gb,vcpuCount:deployment.vcpus}};
  verifyPod(pod);
  // Runpod's Pod gpu.memory is host RAM; 32 GB is valid with a 96 GB GPU.
  verifyPod({...pod,gpu:{...pod.gpu,memory:64,vcpuCount:deployment.vcpus*2}});
  for(const wrong of [{...pod,cost:2.52},{...pod,cost:undefined},{...pod,gpu:{...pod.gpu,id:GPU+' MIG 2g.48gb'}},
    {...pod,gpu:{...pod.gpu,count:2}},{...pod,gpu:{...pod.gpu,vcpuCount:deployment.vcpus-1}},
    {...pod,gpu:{...pod.gpu,memory:undefined}}])assert.throws(()=>verifyPod(wrong));
});
test('remote hardware validation requires a full device and a valid compatible driver',()=>{
  const profile={gpu:'Fixture GPU',compute_capability:'12.0',device_memory_mib:90000,minimum_driver:[575,51,3]};
  for(const memory of [95000,98000])for(const driver of ['575.51.03','575.52.01','580.0.0'])
    assert.equal(verifyDevice(`Fixture GPU, 12.0, ${memory}, ${driver}`,profile).memory_mib,memory);
  for(const row of ['Fixture GPU, 12.0, 48000, 580.0.0','Fixture GPU MIG, 12.0, 98000, 580.0.0',
    'Fixture GPU, 9.0, 98000, 580.0.0','Fixture GPU, 12.0, N/A, 580.0.0',
    'Fixture GPU, 12.0, 98000, N/A','Fixture GPU, 12.0, 98000, 574.99.99',
    'Fixture GPU, 12.0, 98000, 575.51.02','Fixture GPU, 12.0, 98000, 580.0.0\nFixture GPU, 12.0, 98000, 580.0.0'])
    assert.throws(()=>verifyDevice(row,profile));
  assert.equal(deployment.gpu,'NVIDIA RTX PRO 6000 Blackwell Server Edition');
  assert.equal(verifyDevice(`${deployment.gpu}, 12.0, 97280, 580.0.0`).compute_capability,'12.0');
});
test('lost create response is reconciled after restart without a second rental or deadline reset',async()=>{
  const api=new MockApi();api.lost=true;let durable;
  const rental=new Rental(api,initial(),s=>{durable=structuredClone(s);},()=>100);
  await rental.volume();await assert.rejects(rental.pod('public'),/connection lost/);
  assert.equal(durable.pod_requested,true);assert.equal(durable.pod_id,undefined);
  const resumed=new Rental(api,durable,s=>{durable=structuredClone(s);},()=>500);
  await resumed.pod('public');assert.equal(durable.pod_id,'pod1');
  assert.equal(api.calls.filter(c=>c.method==='POST'&&c.uri==='/pods').length,1);
  assert.equal(durable.deadline_ms,7200000);
  api.resources.push({id:'unrelated',name:'someone-elses-pod'});
  await resumed.terminate();assert.equal(durable.compute_terminated,true);
  assert.deepEqual(api.resources.map(p=>p.id),['unrelated']);
});
test('ambiguous provisioning must not be reported as terminated when no response is visible yet',async()=>{
  const api=new MockApi(),state={...initial(),pod_requested:true};
  const rental=new Rental(api,state,()=>{},()=>0);
  await assert.rejects(rental.pod('public'),/unresolved/);
  await assert.rejects(rental.terminate(),/unresolved/);
  assert.equal(state.compute_terminated,undefined);assert.equal(api.calls.length,0);
});
test('deadlines prevent creation; API errors retain storage and retryable termination',async()=>{
  const api=new MockApi(),rental=new Rental(api,initial(),()=>{},()=>2000000);
  await assert.rejects(rental.volume(),/deadline/);await assert.rejects(rental.pod('public'),/deadline/);
  rental.now=()=>1;await rental.volume();
  assert.equal(api.calls.filter(c=>c.uri==='/pods').length,0);
  api.resources=[{id:'pod1',name:rental.state.name}];rental.state.pod_id='pod1';api.deletionFails=true;
  await assert.rejects(rental.terminate(),/temporary/);assert.equal(rental.state.compute_terminated,undefined);
  await assert.rejects(rental.cleanupVolume(true),/Terminate compute/);
  api.deletionFails=false;await rental.terminate();await rental.cleanupVolume(false);
  assert.equal(api.volumes.length,1);assert.equal(rental.state.volume_deleted,undefined);
  await rental.cleanupVolume(true);assert.equal(api.volumes.length,0);
});
test('independent watchdog enforces setup and hard deadlines regardless of controller progress',()=>{
  const state=initial();
  for(const now of [1,1800000-1])assert.equal(terminationDue(state,now),false);
  assert.equal(terminationDue(state,1800000),true);
  state.setup_complete=true;
  assert.equal(terminationDue(state,1800000),false);
  assert.equal(terminationDue(state,7200000-1),false);
  assert.equal(terminationDue(state,7200000),true);
  assert.equal(terminationDue(state,100,true),true);
});
test('a definitive provisioning rejection cleans up the new empty volume',async()=>{
  const api=new MockApi(),request=api.request.bind(api);
  api.request=async(method,uri,body)=>{
    if(method==='POST'&&uri==='/pods'){const error=new Error('Insufficient balance');error.status=400;throw error;}
    return request(method,uri,body);
  };
  const rental=new Rental(api,initial(),()=>{},()=>100);
  await rental.volume();await assert.rejects(rental.pod('public'),/balance/);
  assert.equal(rental.state.pod_rejected,true);
  await rental.terminate();await rental.cleanupVolume(false);
  assert.equal(api.volumes.length,0);assert.equal(rental.state.compute_terminated,true);
});
test('lost volume response is reconciled and cleaned without another volume create',async()=>{
  const api=new MockApi(),request=api.request.bind(api);
  api.request=async(method,uri,body)=>{
    const result=await request(method,uri,body);
    if(method==='POST'&&uri==='/network-volumes')throw new Error('lost volume response');
    return result;
  };
  const rental=new Rental(api,initial(),()=>{},()=>100);
  await assert.rejects(rental.volume(),/lost volume/);
  await rental.terminate();await rental.cleanupVolume(false);
  assert.equal(api.volumes.length,0);
  assert.equal(api.calls.filter(c=>c.method==='POST'&&c.uri==='/network-volumes').length,1);
});
test('REST handles empty deletions, missing resources, errors and cursor pagination',async()=>{
  const calls=[],api=new Api('test-key',async(url,options)=>{
    calls.push({url,options});
    if(options.method==='DELETE')return new Response(null,{status:204});
    if(url.endsWith('/missing'))return new Response(null,{status:404});
    if(url.endsWith('/denied'))return new Response('secret provider detail',{status:403});
    return Response.json(url.includes('cursor=')?{pods:[{id:'second'}],pagination:{nextCursor:null,hasNextPage:false}}:
      {pods:[{id:'first'}],pagination:{nextCursor:'next page',hasNextPage:true}});
  });
  assert.deepEqual((await api.pods()).map(p=>p.id),['first','second']);
  assert.ok(calls[1].url.includes('cursor=next%20page'));
  assert.equal(await api.request('DELETE','/pods/a'),null);assert.equal(await api.request('GET','/missing'),null);
  await assert.rejects(api.request('GET','/denied'),e=>e.status===403&&!e.message.includes('secret'));
});
