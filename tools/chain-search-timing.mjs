// Sequential, alternating paired repeats of the complete frozen pilot.
import fs from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..'),version=process.argv[2]??'v3';
if(!/^v\d+$/.test(version))throw Error('Usage: VERSION');
const read=async p=>JSON.parse(await fs.readFile(resolve(root,p),'utf8'));
const pilot=await read('research/pilot.json');
const report={version:1,algorithm:version,complete:false,scope:'Three sequential full-pilot pairs, alternating order; shared workstation. Other independent audits may occupy separate CPU cores.',pairs:[],assets:[]};
const write=()=>fs.writeFile(resolve(root,'research/chain-search/timing.json'),JSON.stringify(report,null,2)+'\n');
await write();
for(let repeat=1;repeat<=3;repeat++) {
 const pair={repeat,order:repeat%2?['baseline','graph-2']:['graph-2','baseline'],runs:{}};
 for(const method of pair.order) {
  const legacy=method==='baseline',binary=`build/chain-search/${legacy?'baseline':version}/blitz`,stamp=`research/chain-search/builds/${legacy?'baseline':`candidate-${version}`}.json`;
  const dir=`research/runs/chain-search-timing-${version}-r${repeat}-${method}`;
  await new Promise((yes,no)=>{const c=spawn(resolve(root,binary),['bench','research/pilot.json',`research/chain-search/configs/coverage-${method}.json`,dir,'--build-stamp',stamp,'--minutes','50'],{cwd:root,stdio:'inherit'});c.on('error',no);c.on('close',code=>code===0?yes():no(Error('Incomplete timing batch')));});
  const metadata=await read(dir+'/metadata.json'),summary=await read(dir+'/summary.json');
  if(!summary.complete)throw Error('Incomplete timing batch');
  const original=`research/runs/chain-search-${legacy?'':version+'-'}coverage-${method}`;
  const expected=await read(original+'/metadata.json');if(metadata.run_sha256!==expected.run_sha256)throw Error('Timing workload provenance differs from scored workload');
  pair.runs[method]={dir,run_sha256:metadata.run_sha256,rows:[]};
  for(const a of pilot.assets){const row=await read(`${dir}/rows/${a.id}.json`),reference=await read(`${original}/rows/${a.id}.json`);
   if(!row.complete||row.failed||row.attributes_sha256!==reference.attributes_sha256||row.canonical_attributes_sha256!==reference.canonical_attributes_sha256||row.run_sha256!==metadata.run_sha256)throw Error('Timing changed geometry or failed');
   pair.runs[method].rows.push({id:a.id,seconds:row.generation_seconds,output_sha256:row.output_sha256,attributes_sha256:row.attributes_sha256});
  }
 }
 report.pairs.push(pair);await write();
}
const median=a=>a.toSorted((a,b)=>a-b)[Math.floor(a.length/2)];
report.assets=pilot.assets.map(a=>{const ratios=report.pairs.map(p=>p.runs['graph-2'].rows.find(r=>r.id===a.id).seconds/p.runs.baseline.rows.find(r=>r.id===a.id).seconds);return {id:a.id,ratios,median_ratio:median(ratios),max_ratio:Math.max(...ratios)};});
report.complete=true;report.median_ratio_gate=report.assets.every(a=>a.median_ratio<=4);report.all_repeats_gate=report.assets.every(a=>a.max_ratio<=4);await write();
console.log(JSON.stringify({complete:true,median_ratio_gate:report.median_ratio_gate,all_repeats_gate:report.all_repeats_gate,assets:report.assets},null,2));
