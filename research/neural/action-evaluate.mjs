// Inference/auditing only. Saved model hashes and health are checked before use.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {read,write} from './runpod-api.mjs';
import {actionHealth} from './action-gates.mjs';
export function fullPilotFits(report,remainingMs){
  if(!report.complete||report.runs?.length!==9||!Number.isFinite(remainingMs))return false;
  // The diagnostic has two small assets. A 2x uncertainty factor over the
  // eight/two asset ratio avoids treating it as an accurate corpus ETA.
  const duration=report.runs.reduce((sum,r)=>sum+r.wall_seconds,0)*1000;
  return Number.isFinite(duration)&&duration>0&&duration*8+60000<remainingMs;
}
export function savedActionModels(){
  const evidence=fileURLToPath(new URL('./evidence/action-curriculum-v2',import.meta.url)),models=[];
  for(const seed of [101,202,303]){
    const model=evidence+`/seed-${seed}-step-8192.blzn`,health=read(evidence+`/health-${seed}.json`);
    if(!actionHealth(health)||health.step!==8192||crypto.createHash('sha256').update(fs.readFileSync(model)).digest('hex')!==health.model_sha256)throw new Error('Saved evaluation model is unverified');
    models.push(model);
  }
  return models;
}
export async function evaluateSaved({root,execute,deadline,phase}){
  const models=savedActionModels();
  const result={complete:false,training_steps:0,quality_proven:false,generalization_gate_passed:false};
  const run=async scenario=>{
    phase('audit');write(root+'/progress.json',{phase:'audit',scenario,at:Date.now()});
    const minutes=Math.min(45,(deadline-Date.now()-20000)/60000);if(minutes<1)throw new Error('Insufficient evaluation time');
    const out=root+'/'+scenario;
    await execute('node',['research/neural/action-pilot.mjs',out,...models,'--scenario',scenario,'--neural-confirmation','gpu','--minutes',String(minutes),'--action-trials','8','--action-batch','32','--gpu-memory-mib','16384'],root+'/'+scenario+'.log',minutes+.1);
    return read(out+'/report.json');
  };
  result.diagnostic=await run('diagnostic');
  if(!result.diagnostic.complete){result.stop_reason='incomplete_diagnostic';return result;}
  result.complete=true;
  if(fullPilotFits(result.diagnostic,deadline-Date.now())){
    result.pilot=await run('pilot');result.stop_reason=result.pilot.complete?'comparison_complete':'incomplete_full_pilot';
  }else result.stop_reason='diagnostic_complete_insufficient_full_pilot_time';
  result.next_phase='inspect_comparison_before_training';return result;
}
