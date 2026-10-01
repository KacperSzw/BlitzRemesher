// Deterministic test contracts only. No corpus preparation or learning cycle.
export const coreValidationAuthorization = Object.freeze({
  id: 'core-validation-2026-10-01',
  cap_usd: 1.25,
});
export const coreValidationProfiles = Object.freeze([
  'hardware-validation-ada16',
  'hardware-validation-l40s',
]);
export function coreValidationBudget({ states = [], rate, now = Date.now(), minutes = 35 }) {
  if (!Number.isFinite(rate) || rate <= 0 || rate > 1.1 || minutes !== 35 || !Number.isFinite(now))
    throw new Error('core validation requires a bounded 35-minute compatible GPU');
  let prior = 0;
  const names = new Set();
  for (const state of states) {
    if (state.experiment !== 'core-validation' || names.has(state.name)) continue;
    if (state.budget?.authorization?.id !== coreValidationAuthorization.id)
      throw new Error('reconcile unknown core validation authorization');
    const end = state.compute_terminated ? state.terminated_at : now,
      chargedRate = state.deployment?.gpu_hourly_usd_cap;
    if (
      !state.name ||
      !Number.isFinite(state.started_at) ||
      !Number.isFinite(end) ||
      end < state.started_at ||
      !Number.isFinite(chargedRate) ||
      chargedRate <= 0 ||
      chargedRate > 1.1
    )
      throw new Error('reconcile incomplete core validation ledger');
    names.add(state.name);
    prior += ((end - state.started_at) / 3600000) * (chargedRate + 0.01);
  }
  const rental = (minutes / 60) * (rate + 0.01),
    reserve = 0.1,
    total = prior + rental + reserve;
  if (total > coreValidationAuthorization.cap_usd)
    throw new Error('new core validation grant would be exceeded');
  return {
    authorization: coreValidationAuthorization,
    cap_usd: coreValidationAuthorization.cap_usd,
    prior_assumed_usd: prior,
    maximum_rental_usd: rental,
    reserve_usd: reserve,
    maximum_total_usd: total,
    minutes,
  };
}
export function coreValidationSteps(directory) {
  return [
    {
      name: 'ctest',
      command: 'ctest',
      args: [
        '--test-dir',
        'build/neural',
        '--output-on-failure',
        '--timeout',
        '120',
        '--output-junit',
        directory + '/ctest.xml',
      ],
      maximum: 300000,
    },
    {
      name: 'vulkan-validation',
      command: 'build/neural/blitz-neural-vulkan-tests',
      args: [],
      maximum: 120000,
      env: { VK_INSTANCE_LAYERS: 'VK_LAYER_KHRONOS_validation' },
    },
    ...['action-gpu', 'vulkan'].map((name) => ({
      name: name + '-memcheck',
      command: '/usr/local/cuda/bin/compute-sanitizer',
      args: [
        '--tool',
        'memcheck',
        '--error-exitcode',
        '1',
        'build/neural/blitz-neural-' + name + '-tests',
        ...(name === 'vulkan' ? ['--memcheck'] : []),
      ],
      maximum: 120000,
      env: { CUDA_MODULE_LOADING: 'EAGER', CUDA_MODULE_DATA_LOADING: 'EAGER' },
    })),
  ];
}
export function validateCoreTestLog(name, log) {
  if (name === 'ctest' && (/\*\*\*Skipped|Not Run/.test(log) || !log.includes('100% tests passed')))
    throw new Error('CTest incomplete or skipped required contracts');
  if (name === 'vulkan-validation' && /Validation Error|VUID-|SYNC-HAZARD|was not found/.test(log))
    throw new Error('Vulkan validation error or missing layer');
  if (name.endsWith('-memcheck') && !/ERROR SUMMARY: 0 errors/.test(log))
    throw new Error('missing clean Compute Sanitizer result');
}
