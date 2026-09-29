// One approved, full-GPU deployment; copied with the controller and source bundle.
export const deployment=Object.freeze({
  gpu:'NVIDIA RTX PRO 6000 Blackwell Server Edition',
  catalog_vram_gb:96,device_memory_mib:90000,compute_capability:'12.0',cuda_architecture:120,
  minimum_driver:[575,51,3],minimum_cuda:'12.9',
  host_ram_gb:32,vcpus:8,gpu_hourly_usd_cap:2.50,
  container_disk_gb:50,network_volume_gb:20
});

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
