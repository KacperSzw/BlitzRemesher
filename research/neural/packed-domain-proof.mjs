// The policy and condition that aborted expanded-pretraining-02 are immutable
// regression inputs. This proof never counts toward the two-hour training run.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {read,write} from './runpod-api.mjs';
const fixture=fileURLToPath(new URL('evidence/packed-domain-failure',import.meta.url));
export async function replayPackedDomain(ctx,memory=16384){
  const outcome=read(fixture+'/outcome.json'),model=fixture+'/policy.blzn';
  if(crypto.createHash('sha256').update(fs.readFileSync(model)).digest('hex')!==outcome.policy_sha256)throw new Error('Packed failure policy changed');
  const report={complete:false,score:null,policy_sha256:outcome.policy_sha256,rows:[]};
  for(const batch of [1,4,8]){
    const directory=ctx.root+'/packed-replay-'+batch;
    const code=await ctx.execute('blitz-neural-placement-prepare',['ph_wooden_display_shelves_01',directory,
      '--states','16','--pool','4','--pixels','128','--seed','1495','--minutes','2','--gpu-memory-mib',String(memory),
      '--raster-backend','vulkan','--vertex-storage','packed','--candidate-batch',String(batch),'--audit-mode','sparse',
      '--training-profile','coverage','--corpus','research/neural/corpus-v2/corpus.json','--training-selection','research/neural/corpus-v2/training.json',
      '--simplifier-seed','on','--retained','0.25','--model',model],directory+'.log',2.1);
    const index=read(directory+'/index.json');report.rows.push({batch,code,index});write(ctx.root+'/packed-replay.json',report);
    if(code!==0||!index.complete||!index.reference_confirmed||!index.seed.accepted||index.states!==16||!index.invalid_candidates)throw new Error('Packed failure replay did not complete the rejected-candidate boundary');
  }
  if(new Set(report.rows.map(r=>r.index.sha256)).size!==1||new Set(report.rows.map(r=>r.index.episode_sha256)).size!==1)throw new Error('Packed replay differs between serial and batched candidates');
  report.complete=true;write(ctx.root+'/packed-replay.json',report);return report;
}
export async function soakPackedDomain(ctx,{memory=16384,minutes=20}={}){
  if(!Number.isInteger(minutes)||minutes<1||minutes>20)throw new Error('Packed validation soak outside 1..20 minutes');
  const directory=ctx.root+'/packed-soak';
  const code=await ctx.execute('blitz-neural-cycle',[directory,'--duration-minutes',String(minutes),'--finalize-minutes','2','--minutes',String(minutes+2),
    '--states','16','--updates','128','--batch','512','--seed','101','--gpu-memory-mib',String(memory),
    '--raster-backend','vulkan','--vertex-storage','packed','--data-storage','compact','--training-profile','coverage',
    '--candidate-batch','4','--update-backend','fused','--workers','2','--hidden-width','64','--checkpoint-seconds','60',
    '--episode-seeds','on','--quality','on','--initialize',fixture+'/policy.blzn',
    '--corpus','research/neural/corpus-v2/corpus.json','--training-selection','research/neural/corpus-v2/training.json',
    '--curriculum','research/neural/corpus-v2/curriculum.json','--mesh-cache',ctx.root+'/mesh-cache'],directory+'.log',minutes+2.1);
  const report=read(directory+'/report.json');
  if(code!==0||!report.complete||report.learning_elapsed_ms<minutes*60000||report.failed_conditions_count||report.coverage_quality_failed||report.next_condition<1395)throw new Error('Packed soak did not pass the former failing condition with a complete learning window');
  return report;
}
