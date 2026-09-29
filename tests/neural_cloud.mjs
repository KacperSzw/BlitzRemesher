import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Api,Rental,GPU,IMAGE,chooseQuote,podRequest,verifyPod,terminationDue} from '../research/neural/runpod-api.mjs';
const gpu={id:GPU,memory:32,secure:true,price:{secure:.8},dataCenters:[{id:'EU-1',availability:'HIGH'}]};
const centers=[{id:'EU-1',networkVolumeTypes:['STANDARD']}];
const initial=()=>({name:'test-unique',quote:chooseQuote([gpu],centers),setup_deadline_ms:1800000,deadline_ms:7200000});
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
  for(const invalid of [{...gpu,id:'RTX 4090'},{...gpu,memory:24},{...gpu,secure:false},
    {...gpu,price:{secure:1.01}},{...gpu,price:{secure:NaN}},{...gpu,dataCenters:[{id:'EU-1',availability:'NONE'}]}])
    assert.throws(()=>chooseQuote([invalid],centers));
  assert.throws(()=>chooseQuote([gpu],[]));
  assert.equal(chooseQuote([gpu],centers).gpu_hourly_usd,.8);
});
test('Pod request is pinned and forwards only the public key',()=>{
  const s={...initial(),volume_id:'volume1'},body=podRequest(s,'ssh-ed25519 public-test\n');
  assert.equal(body.image,IMAGE);assert.match(body.image,/@sha256:[a-f0-9]{64}$/);
  assert.equal(body.gpu.count,1);assert.equal(body.cloud,'SECURE');
  assert.deepEqual(body.env,{PUBLIC_KEY:'ssh-ed25519 public-test'});
  assert.deepEqual(body.ports,['22/tcp']);assert.equal(body.mounts.network[0].volumeId,'volume1');
});
test('actual allocation must satisfy hardware and rate caps',()=>{
  const pod={cloud:'SECURE',cost:.91,gpu:{id:GPU,count:1,memory:32,vcpuCount:8}};
  verifyPod(pod);
  for(const wrong of [{...pod,cost:2},{...pod,cost:undefined},{...pod,gpu:{...pod.gpu,count:2}},{...pod,gpu:{...pod.gpu,vcpuCount:4}},
    {...pod,gpu:{...pod.gpu,memory:undefined}}])assert.throws(()=>verifyPod(wrong));
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
