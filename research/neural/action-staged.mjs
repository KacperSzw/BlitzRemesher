// A complete full-settings two-asset check precedes any new optimizer updates.
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {read,write} from './runpod-api.mjs';
import {generalize} from './action-generalize.mjs';
import {savedActionModels} from './action-evaluate.mjs';
import {auditRowsHealthy} from './action-gates.mjs';
export async function staged({root,execute,deadline,phase}){
  const baseline=root+'/baseline',manifest=fileURLToPath(new URL('./action-diagnostic.json',import.meta.url));
  const models=savedActionModels();phase('audit');
  await execute('blitz',['bench',manifest,'research/neural/action-pilot.json',baseline,'--neural-model',models[0],'--neural-confirmation','gpu','--action-trials','8','--action-batch','32','--gpu-memory-mib','16384','--minutes','10'],root+'/baseline.log',10.2);
  const summary=read(baseline+'/summary.json'),assets=read(manifest).assets;
  const rows=assets.map(a=>baseline+'/rows/'+a.id+'.json').filter(f=>fs.existsSync(f)).map(read);
  const healthy=summary.complete&&Number.isFinite(summary.seconds)&&summary.seconds>0&&auditRowsHealthy(rows,assets);
  write(root+'/baseline-health.json',{healthy,summary,rows:rows.map(r=>({id:r.id,seconds:r.seconds,ratio:r.ratio,neural:r.neural}))});
  if(!healthy)return {complete:false,training_steps:0,quality_proven:false,stop_reason:'incomplete_or_unhealthy_full_settings_baseline'};
  // The first paid milestone is verified labels and healthy optimizer updates.
  // Nine full comparisons cost over an hour on the measured H200 workload;
  // do not make that later pilot a prerequisite for this bounded screening.
  if(Date.now()+5*60000>=deadline)
    return {complete:false,training_steps:0,quality_proven:false,stop_reason:'insufficient_time_for_training'};
  return generalize({root,execute,deadline,phase,diagnostic:true,screening:true,refresh:true});
}
