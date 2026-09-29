// Shared contracts for the sustained runner and bounded cloud calibration.
export function trainingDeadline(started,latest,minutes){
  if(!Number.isFinite(started)||!Number.isFinite(latest)||!Number.isFinite(minutes)||minutes<=0||latest<=started)
    throw new Error('Invalid training window');
  return Math.min(latest,started+minutes*60000);
}
export function initialization(argument) {
  if(!argument)throw new Error('Specify --from-scratch or an initial model');
  return argument==='--from-scratch'?null:argument;
}
export function trainerArguments(dataset,run,initialize,config,steps,minutes) {
  const args=[dataset,run,'--steps',String(steps),'--segment-minutes',String(minutes),
    '--checkpoint-every',String(config.checkpoint_every),'--core',String(config.core),
    '--batch',String(config.batch),'--workers',String(config.workers??2),
    '--gpu-memory-mib',String(config.gpu_memory_mib??5120)];
  if(initialize!==null)args.push('--initialize',initialize);
  return args;
}
export function checkpointHealthy(h) {
  return h?.finite===true&&h.restored===true&&h.optimizer_restored===true&&
    Number.isFinite(h.native_max_abs_error)&&h.native_max_abs_error<=2e-4&&
    Number.isFinite(h.parameter_change)&&h.parameter_change>0&&
    Number.isFinite(h.gradient_norm)&&h.gradient_norm>0;
}
export function trainingWindow(samples) {
  const window=[];
  for(const sample of [...samples].reverse()){
    if(sample.phase!=='training'){if(window.length)break;continue;}
    if(window.length&&Date.parse(window[0].at)-Date.parse(sample.at)>2500)break;
    window.unshift(sample);if(window.length===62)break;
  }
  const sorted=window.map(s=>s.gpu).sort((a,b)=>a-b);
  const mean=values=>values.length?values.reduce((sum,v)=>sum+v,0)/values.length:0;
  return {samples:window.length,seconds:window.length?(Date.parse(window.at(-1).at)-Date.parse(window[0].at))/1000:0,
    mean:mean(sorted),p10:sorted[Math.floor(sorted.length*.1)]??0,
    minimum:sorted[0]??0,maximum:sorted.at(-1)??0,
    mean_power_w:mean(window.map(s=>s.power_w)),peak_device_memory_mib:Math.max(0,...window.map(s=>s.memory_mib))};
}
export function selectCalibration(trials) {
  const passing=trials.filter(t=>checkpointHealthy(t.health)&&t.measured_steps>=16&&
    Number.isFinite(t.vertices_per_second)&&t.vertices_per_second>0);
  if(!passing.length)throw new Error('No calibration candidate passed checkpoint and parity checks');
  return passing.sort((a,b)=>b.vertices_per_second-a.vertices_per_second)[0];
}
