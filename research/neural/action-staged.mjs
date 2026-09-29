// A complete full-settings two-asset check precedes any new optimizer updates.
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {read,write} from './runpod-api.mjs';
import {generalize} from './action-generalize.mjs';
import {savedActionModels} from './action-evaluate.mjs';
export async function staged({root,execute,deadline,phase}){
  const baseline=root+'/baseline',manifest=fileURLToPath(new URL('./action-diagnostic.json',import.meta.url));
  const models=savedActionModels();phase('audit');
  await execute('blitz',['bench',manifest,'research/neural/action-pilot.json',baseline,'--neural-model',models[0],'--neural-confirmation','gpu','--action-trials','8','--action-batch','32','--gpu-memory-mib','16384','--minutes','10'],root+'/baseline.log',10.2);
  const summary=read(baseline+'/summary.json'),assets=read(manifest).assets;
  const rows=assets.map(a=>baseline+'/rows/'+a.id+'.json').filter(f=>fs.existsSync(f)).map(read);
  const healthy=summary.complete&&Number.isFinite(summary.seconds)&&summary.seconds>0&&rows.length===assets.length&&rows.every(r=>r.complete&&!r.failed&&r.neural&&!r.neural.resource_failures&&!r.neural.confirmation_resources&&!r.neural.confirmation_nonfinite&&!r.neural.confirmation_disagreements);
  write(root+'/baseline-health.json',{healthy,summary,rows:rows.map(r=>({id:r.id,seconds:r.seconds,ratio:r.ratio,neural:r.neural}))});
  if(!healthy)return {complete:false,training_steps:0,quality_proven:false,stop_reason:'incomplete_or_unhealthy_full_settings_baseline'};
  // Reserve two minutes for training/collection and at least one diagnostic
  // duration at each of the nine rankings, plus bounded label collection.
  if(Date.now()+Math.max(300000,summary.seconds*9000)+20*60000>=deadline)
    return {complete:false,training_steps:0,quality_proven:false,stop_reason:'insufficient_time_for_training_and_matched_audit'};
  return generalize({root,execute,deadline,phase,diagnostic:true,refresh:true});
}
