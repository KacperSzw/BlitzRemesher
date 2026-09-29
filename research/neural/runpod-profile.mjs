// Full-GPU profiles, copied with the controller and source bundle. FP64 devices
// are bounded follow-ups to the measured Blackwell audit workload.
const blackwell={
  id:'blackwell',
  gpu:'NVIDIA RTX PRO 6000 Blackwell Server Edition',
  catalog_vram_gb:96,device_memory_mib:90000,compute_capability:'12.0',cuda_architecture:120,
  minimum_driver:[575,51,3],minimum_cuda:'12.9',
  host_ram_gb:32,vcpus:16,gpu_hourly_usd_cap:2.50,
  container_disk_gb:50,network_volume_gb:20,
  setup_minutes:30,training_minutes:120,collection_minutes:10,
  staged_rental_minutes:150,staged_experiment_minutes:110
};
export const profiles=Object.freeze({
  blackwell:Object.freeze(blackwell),
  h100:Object.freeze({...blackwell,id:'h100',gpu:'NVIDIA H100 NVL',catalog_vram_gb:94,
    compute_capability:'9.0',cuda_architecture:90,gpu_hourly_usd_cap:3.25,
    staged_rental_minutes:90,staged_experiment_minutes:50}),
  a100:Object.freeze({...blackwell,id:'a100',gpu:'NVIDIA A100 80GB PCIe',catalog_vram_gb:80,
    device_memory_mib:76000,compute_capability:'8.0',cuda_architecture:80,gpu_hourly_usd_cap:1.65,
    staged_rental_minutes:150,staged_experiment_minutes:110}),
  h200:Object.freeze({...blackwell,id:'h200',gpu:'NVIDIA H200',catalog_vram_gb:141,
    device_memory_mib:130000,compute_capability:'9.0',cuda_architecture:90,gpu_hourly_usd_cap:4.60,
    staged_rental_minutes:60,staged_experiment_minutes:40})
});
const selected=process.env.BLITZ_RUNPOD_PROFILE??'blackwell';
if(!Object.hasOwn(profiles,selected))throw new Error('Unknown bounded RunPod profile');
export const deployment=profiles[selected];

export function verifyDevice(csv,profile=deployment){
  const rows=csv.trim().split('\n');
  if(rows.length!==1)throw new Error('Expected exactly one full GPU');
  const [name,cap,memory,driver]=rows[0].split(',').map(s=>s.trim());
  const version=/^\d+\.\d+\.\d+$/.test(driver??'')?driver.split('.').map(Number):[];
  let older=version.length!==3;
  for(let i=0;!older&&i<3;i++){
    if(version[i]!==profile.minimum_driver[i]){older=version[i]<profile.minimum_driver[i];break;}
  }
  if(name!==profile.gpu||cap!==profile.compute_capability||!Number.isFinite(Number(memory))||
    Number(memory)<profile.device_memory_mib||older)
    throw new Error('GPU or driver does not satisfy the approved full-GPU deployment profile');
  return {name,compute_capability:cap,memory_mib:Number(memory),driver};
}
