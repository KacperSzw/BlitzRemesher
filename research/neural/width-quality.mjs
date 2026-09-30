// Quality comparisons are separate from the fixed-label throughput matrix.
// A partial three-seed comparison never selects a larger network.
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {read,write} from './runpod-api.mjs';
export function widthDecision(rows,seeds,minimumReduction=.05,maximumTimeRatio=2){
  if(seeds.length!==3||new Set(seeds).size!==3||!Number.isFinite(minimumReduction)||minimumReduction<0||minimumReduction>1||!Number.isFinite(maximumTimeRatio)||maximumTimeRatio<=0)throw new Error('invalid three-seed quality gate');
  const result={selected_width:64,passed:false,score:null,comparisons:[]};
  const valid=a=>a?.complete===true&&a.rows?.length&&a.rows.every(r=>r.status==='complete'&&!r.neural?.resource_failures&&!r.neural?.confirmation_nonfinite&&r.lods?.length>1&&r.lods.slice(1).every(l=>l.source?.complete&&l.source?.passed&&l.adjacent?.complete&&l.adjacent?.passed));
  const metrics=a=>a.rows.filter(r=>r.ranking==='learned').map(r=>({asset:r.asset,ratio:r.lods.at(-1).triangles/r.lods[0].triangles,seconds:r.seconds})).sort((a,b)=>a.asset.localeCompare(b.asset));
  for(const width of [128,256]){
    const comparison={width,passed:true,seeds:[]};
    for(const seed of seeds){const base=rows.find(r=>r.seed===seed&&r.width===64)?.audit,next=rows.find(r=>r.seed===seed&&r.width===width)?.audit;
      if(!valid(base)||!valid(next)||base.manifest_sha256!==next.manifest_sha256||base.settings_sha256!==next.settings_sha256){comparison.passed=false;comparison.seeds.push({seed,complete:false});continue;}
      const a=metrics(base),b=metrics(next);if(!a.length||JSON.stringify(a.map(r=>r.asset))!==JSON.stringify(b.map(r=>r.asset)))throw new Error('width comparison assets differ');
      const mean=rs=>rs.reduce((n,r)=>n+r.ratio,0)/rs.length,sum=rs=>rs.reduce((n,r)=>n+r.seconds,0);
      const reduction=1-mean(b)/mean(a),time_ratio=sum(b)/sum(a),passed=Number.isFinite(reduction)&&Number.isFinite(time_ratio)&&reduction>=minimumReduction&&time_ratio<=maximumTimeRatio;
      comparison.seeds.push({seed,complete:true,reduction,time_ratio,passed});comparison.passed&&=passed;
    }
    result.comparisons.push(comparison);if(comparison.passed&&!result.passed){result.selected_width=width;result.passed=true;}
  }
  return result;
}
async function main(){
  const [input,directory,updatesText='128']=process.argv.slice(2),root=path.resolve(directory??''),updates=Number(updatesText);
  if(!input||!directory||![32,128,512,2048].includes(updates)||fs.existsSync(root))throw new Error('width-quality.mjs ABLATION_DIRECTORY NEW_OUTPUT [UPDATES]');
  fs.mkdirSync(root,{recursive:true});const started=Date.now(),deadline=Number(process.env.BLITZ_VALIDATION_DEADLINE??started+600000),seeds=[101,211,307];
  const report={complete:false,score:null,updates,rows:[],gate:widthDecision([],seeds)};let active,stop=false;
  for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{stop=true;active?.kill('SIGTERM');});
  const settings={...read('research/neural/refactor-smoke.json'),profile:'coverage'};write(root+'/settings.json',settings);
  try{for(const seed of seeds)for(const width of [64,128,256]){
    if(stop||Date.now()+5000>=deadline)throw new Error('quality comparison deadline');const label=`seed${seed}-w${width}-u${updates}`,cycle=read(path.resolve(input,label,'report.json'));
    if(!cycle.complete)throw new Error('incomplete learning ablation '+label);const output=root+'/'+label+'.json',fd=fs.openSync(root+'/'+label+'.log','w');
    active=spawn('build/neural/blitz-neural-cycle',['--audit-model','research/neural/corpus-v2/validation.json',path.resolve(input,label,cycle.checkpoint,'model.blzn'),root+'/settings.json',output],{stdio:['ignore',fd,fd]});let hard;
    const timer=setTimeout(()=>{active?.kill('SIGTERM');hard=setTimeout(()=>active?.kill('SIGKILL'),3000);},deadline-Date.now()-3000);
    try{const code=await new Promise((resolve,reject)=>{active.once('error',reject);active.once('close',resolve);});report.rows.push({seed,width,code,audit:fs.existsSync(output)?read(output):null});write(root+'/report.json',report);if(code)throw new Error('quality audit failed '+label);}
    finally{clearTimeout(timer);clearTimeout(hard);fs.closeSync(fd);active=undefined;}
  }report.complete=report.rows.length===9&&report.rows.every(r=>r.audit?.complete===true);}
  catch(error){report.error=String(error);process.exitCode=1;}
  finally{report.gate=widthDecision(report.rows,seeds);report.seconds=(Date.now()-started)/1000;write(root+'/report.json',report);}
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
