// Full-GPU profiles, copied with the controller and source bundle. FP64 devices
// are bounded follow-ups to the measured Blackwell audit workload.
const blackwell = {
  id: 'blackwell',
  gpu: 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
  catalog_vram_gb: 96,
  device_memory_mib: 90000,
  compute_capability: '12.0',
  cuda_architecture: 120,
  minimum_driver: [575, 51, 3],
  minimum_cuda: '12.9',
  host_ram_gb: 32,
  vcpus: 16,
  gpu_hourly_usd_cap: 2.5,
  container_disk_gb: 50,
  network_volume_gb: 20,
  setup_minutes: 30,
  training_minutes: 120,
  collection_minutes: 10,
  staged_rental_minutes: 150,
  staged_experiment_minutes: 110,
};
export const profiles = Object.freeze({
  'pipeline-validation-l40s': Object.freeze({
    ...blackwell,
    id: 'pipeline-validation-l40s',
    gpu: 'NVIDIA L40S',
    catalog_vram_gb: 48,
    device_memory_mib: 45000,
    compute_capability: '8.9',
    cuda_architecture: 89,
    host_ram_gb: 32,
    vcpus: 8,
    gpu_hourly_usd_cap: 1.1,
    setup_minutes: 20,
    training_minutes: 30,
    collection_minutes: 10,
    graphics: true,
  }),
  blackwell: Object.freeze(blackwell),
  'gpu-refactor': Object.freeze({
    ...blackwell,
    id: 'gpu-refactor',
    setup_minutes: 20,
    training_minutes: 150,
    collection_minutes: 10,
    gpu_hourly_usd_cap: 2.1,
    graphics: true,
  }),
  'coverage-pretrain-4090': Object.freeze({
    ...blackwell,
    id: 'coverage-pretrain-4090',
    gpu: 'NVIDIA GeForce RTX 4090',
    catalog_vram_gb: 24,
    device_memory_mib: 22000,
    compute_capability: '8.9',
    cuda_architecture: 89,
    host_ram_gb: 32,
    vcpus: 8,
    gpu_hourly_usd_cap: 0.8,
    setup_minutes: 20,
    training_minutes: 150,
    collection_minutes: 10,
    graphics: true,
  }),
  'coverage-pretrain-l40s': Object.freeze({
    ...blackwell,
    id: 'coverage-pretrain-l40s',
    network_volume_gb: 80,
    gpu: 'NVIDIA L40S',
    catalog_vram_gb: 48,
    device_memory_mib: 45000,
    compute_capability: '8.9',
    cuda_architecture: 89,
    host_ram_gb: 32,
    vcpus: 8,
    gpu_hourly_usd_cap: 1.1,
    setup_minutes: 20,
    training_minutes: 150,
    collection_minutes: 10,
    graphics: true,
  }),
  'coverage-pretrain-4000': Object.freeze({
    ...blackwell,
    id: 'coverage-pretrain-4000',
    gpu: 'NVIDIA RTX PRO 4000 Blackwell',
    catalog_vram_gb: 24,
    device_memory_mib: 22000,
    host_ram_gb: 32,
    vcpus: 8,
    gpu_hourly_usd_cap: 0.6,
    setup_minutes: 20,
    training_minutes: 150,
    collection_minutes: 10,
    graphics: true,
  }),
  'hardware-validation': Object.freeze({
    ...blackwell,
    id: 'hardware-validation',
    setup_minutes: 20,
    training_minutes: 10,
    collection_minutes: 5,
    graphics: true,
  }),
  'hardware-validation-small': Object.freeze({
    ...blackwell,
    id: 'hardware-validation-small',
    gpu: 'NVIDIA RTX PRO 4000 Blackwell',
    catalog_vram_gb: 24,
    device_memory_mib: 22000,
    host_ram_gb: 16,
    vcpus: 4,
    gpu_hourly_usd_cap: 0.6,
    setup_minutes: 20,
    training_minutes: 10,
    collection_minutes: 5,
    graphics: true,
  }),
  'hardware-validation-ada': Object.freeze({
    ...blackwell,
    id: 'hardware-validation-ada',
    gpu: 'NVIDIA GeForce RTX 4090',
    catalog_vram_gb: 24,
    device_memory_mib: 22000,
    compute_capability: '8.9',
    cuda_architecture: 89,
    host_ram_gb: 16,
    vcpus: 4,
    gpu_hourly_usd_cap: 0.8,
    setup_minutes: 20,
    training_minutes: 10,
    collection_minutes: 5,
    graphics: true,
  }),
  'hardware-validation-ada16': Object.freeze({
    ...blackwell,
    id: 'hardware-validation-ada16',
    gpu: 'NVIDIA RTX 2000 Ada Generation',
    catalog_vram_gb: 16,
    device_memory_mib: 14000,
    compute_capability: '8.9',
    cuda_architecture: 89,
    host_ram_gb: 16,
    vcpus: 4,
    gpu_hourly_usd_cap: 0.3,
    setup_minutes: 20,
    training_minutes: 10,
    collection_minutes: 5,
    graphics: true,
  }),
  'hardware-validation-l40s': Object.freeze({
    ...blackwell,
    id: 'hardware-validation-l40s',
    gpu: 'NVIDIA L40S',
    catalog_vram_gb: 48,
    device_memory_mib: 45000,
    compute_capability: '8.9',
    cuda_architecture: 89,
    host_ram_gb: 16,
    vcpus: 4,
    gpu_hourly_usd_cap: 1.1,
    setup_minutes: 20,
    training_minutes: 10,
    collection_minutes: 5,
    graphics: true,
  }),
  h100: Object.freeze({
    ...blackwell,
    id: 'h100',
    gpu: 'NVIDIA H100 NVL',
    catalog_vram_gb: 94,
    compute_capability: '9.0',
    cuda_architecture: 90,
    gpu_hourly_usd_cap: 3.25,
    staged_rental_minutes: 90,
    staged_experiment_minutes: 50,
  }),
  a100: Object.freeze({
    ...blackwell,
    id: 'a100',
    gpu: 'NVIDIA A100 80GB PCIe',
    catalog_vram_gb: 80,
    device_memory_mib: 76000,
    compute_capability: '8.0',
    cuda_architecture: 80,
    gpu_hourly_usd_cap: 1.65,
    staged_rental_minutes: 150,
    staged_experiment_minutes: 110,
  }),
  h200: Object.freeze({
    ...blackwell,
    id: 'h200',
    gpu: 'NVIDIA H200',
    catalog_vram_gb: 141,
    device_memory_mib: 130000,
    compute_capability: '9.0',
    cuda_architecture: 90,
    gpu_hourly_usd_cap: 4.6,
    staged_rental_minutes: 60,
    staged_experiment_minutes: 40,
  }),
});
const selected =
  process.env.BLITZ_RUNPOD_PROFILE ??
  (process.argv[2] === 'prepare-core-validation' ? 'hardware-validation-ada16' : 'blackwell');
if (!Object.hasOwn(profiles, selected)) throw new Error('Unknown bounded RunPod profile');
export const deployment = profiles[selected];

export function verifyDevice(csv, profile = deployment) {
  const rows = csv.trim().split('\n');
  if (rows.length !== 1) throw new Error('Expected exactly one full GPU');
  const [name, cap, memory, driver] = rows[0].split(',').map((s) => s.trim());
  const version = /^\d+\.\d+\.\d+$/.test(driver ?? '') ? driver.split('.').map(Number) : [];
  let older = version.length !== 3;
  for (let i = 0; !older && i < 3; i++) {
    if (version[i] !== profile.minimum_driver[i]) {
      older = version[i] < profile.minimum_driver[i];
      break;
    }
  }
  if (
    name !== profile.gpu ||
    cap !== profile.compute_capability ||
    !Number.isFinite(Number(memory)) ||
    Number(memory) < profile.device_memory_mib ||
    older
  )
    throw new Error('GPU or driver does not satisfy the approved full-GPU deployment profile');
  return { name, compute_capability: cap, memory_mib: Number(memory), driver };
}
