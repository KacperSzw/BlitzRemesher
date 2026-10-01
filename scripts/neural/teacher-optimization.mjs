// This grant covers the approved teacher optimization experiment, not final training.
import { auditTermination, compareAudits } from './quality-metrics.mjs';

export const teacherOptimizationAuthorization = Object.freeze({
  id: 'teacher-optimization-2026-10-01',
  cap_usd: 3.5,
});
export const teacherOptimizationMinutes = Object.freeze({
  setup: 20,
  teacher: 20,
  learning: 42,
  quality: 50,
  collection: 8,
});
export const teacherOptimizationProfiles = Object.freeze([
  'teacher-optimization-a40',
  'teacher-optimization-4090',
  'teacher-optimization-l40s',
  'teacher-optimization-a40-hour',
]);
export const teacherOptimizationSeeds = Object.freeze([101, 211, 307]);

export function selectOptimizationAssets({ assets, trainingIds, teacherIds, auditIds }) {
  const training = new Set(trainingIds),
    teachers = new Set(teacherIds),
    selected = new Set([...teachers, ...auditIds]),
    result = [];
  for (const asset of assets) {
    if (!selected.has(asset.id)) continue;
    if (asset.split !== 'development' || (teachers.has(asset.id) && !training.has(asset.id)))
      throw new Error(
        'Teacher fixtures require the training split; audits require development assets',
      );
    result.push(asset);
    selected.delete(asset.id);
  }
  if (selected.size) throw new Error('Missing teacher optimization assets');
  return result;
}

export function teacherOptimizationBudget({ states = [], rate, now = Date.now(), minutes = 140 }) {
  if (
    !Number.isFinite(rate) ||
    rate <= 0 ||
    rate > 1.1 ||
    ![140, 232].includes(minutes) ||
    !Number.isFinite(now)
  )
    throw new Error('teacher optimization requires a bounded compatible GPU rental');
  let prior = 0;
  const names = new Set();
  for (const state of states) {
    if (state.experiment !== 'teacher-optimization' || names.has(state.name)) continue;
    if (state.budget?.authorization?.id !== teacherOptimizationAuthorization.id)
      throw new Error('reconcile unknown teacher optimization authorization');
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
      throw new Error('reconcile incomplete teacher optimization ledger');
    names.add(state.name);
    // Reserve disk charges as well as the maximum GPU rate, including setup failures.
    prior += ((end - state.started_at) / 3600000) * (chargedRate + 0.03);
  }
  const rental = (minutes / 60) * (rate + 0.03),
    reserve = 0.3,
    total = prior + rental + reserve;
  if (total > teacherOptimizationAuthorization.cap_usd)
    throw new Error('teacher optimization grant would be exceeded');
  return {
    authorization: teacherOptimizationAuthorization,
    cap_usd: teacherOptimizationAuthorization.cap_usd,
    prior_assumed_usd: prior,
    maximum_rental_usd: rental,
    reserve_usd: reserve,
    maximum_total_usd: total,
    minutes,
  };
}

export function validateOptimizationRequest(request) {
  if (
    request?.version !== 1 ||
    request.final_training !== false ||
    !/^[a-f0-9]{40}$/.test(request.baseline_revision ?? '') ||
    !/^[a-f0-9]{64}$/.test(request.initialization_sha256 ?? '') ||
    JSON.stringify(request.seeds) !== JSON.stringify(teacherOptimizationSeeds) ||
    ![32, 128, 512, 2048, 8192, 32768, 65536].includes(request.action_trials) ||
    request.action_batch !== 16 ||
    request.gpu_memory_mib !== 16384 ||
    request.workers !== 2 ||
    request.candidate_batch !== 4 ||
    request.learning_minutes !== 5 ||
    request.finalize_minutes !== 2 ||
    (request.capability_minutes !== undefined && request.capability_minutes !== 60) ||
    (request.timeout_diagnostics !== undefined &&
      typeof request.timeout_diagnostics !== 'boolean') ||
    (request.vulkan_icd !== undefined && !['glx', 'egl'].includes(request.vulkan_icd)) ||
    (request.baseline_overlay !== undefined &&
      !['worker-join-v1', 'worker-join-seed-v2'].includes(request.baseline_overlay))
  )
    throw new Error('invalid frozen teacher optimization request');
  return request;
}

export function optimizationCycleArguments(
  request,
  { run, seed, strategy, model, curriculum, capability = false },
) {
  validateOptimizationRequest(request);
  if (!request.seeds.includes(seed) || !['exhaustive', 'coverage-core-first'].includes(strategy))
    throw new Error('unapproved teacher pilot seed or strategy');
  if (
    capability &&
    (request.capability_minutes !== 60 || seed !== 101 || strategy !== 'exhaustive')
  )
    throw new Error('capability run requires the approved exhaustive hour and seed');
  return [
    run,
    '--initialize',
    model,
    '--duration-minutes',
    String(capability ? request.capability_minutes : request.learning_minutes),
    '--finalize-minutes',
    '2',
    '--architecture',
    '4',
    '--hidden-width',
    '64',
    '--training-profile',
    'coverage',
    '--teacher-strategy',
    strategy,
    '--teacher-selection',
    'policy-mixed',
    '--policy-rollout-trials',
    '64',
    '--episode-seeds',
    'on',
    '--simplifier-seeds',
    'on',
    '--states',
    '16',
    '--updates',
    '128',
    '--batch',
    '512',
    '--seed',
    String(seed),
    '--workers',
    String(request.workers),
    '--candidate-batch',
    String(request.candidate_batch),
    '--checkpoint-seconds',
    '60',
    '--update-backend',
    'fused',
    '--raster-backend',
    'vulkan',
    '--vertex-storage',
    'packed',
    '--data-storage',
    'compact',
    '--mask-only-coverage',
    'on',
    '--gpu-memory-mib',
    String(request.gpu_memory_mib),
    '--curriculum',
    curriculum,
    '--corpus',
    'research/neural/corpus-v2/corpus.json',
    '--training-selection',
    'research/neural/corpus-v2/training.json',
    '--quality',
    'off',
  ];
}

export function teacherStrategyGate(pairs, expectedAssets) {
  if (pairs.length !== teacherOptimizationSeeds.length || expectedAssets !== 12)
    throw new Error('three complete seeds and the full frozen development pilot are required');
  const seen = new Set();
  const comparisons = pairs.map(({ seed, exhaustive, candidate }) => {
    if (!teacherOptimizationSeeds.includes(seed) || seen.has(seed))
      throw new Error('duplicate or unexpected quality seed');
    seen.add(seed);
    const result = compareAudits(exhaustive, candidate);
    if (
      result.base.assets.length !== expectedAssets ||
      result.base.categories.length !== 4 ||
      result.base.assets.some((a) => a.triangles.length !== 8)
    )
      throw new Error('full eight-LOD development pilot required');
    return {
      seed,
      ...result,
      termination: [exhaustive, candidate].map(auditTermination),
    };
  });
  return {
    version: 2,
    complete: true,
    matched_budget_nonregression: comparisons.every((c) => c.per_lod_regressions.length === 0),
    passed: comparisons.every(
      (c) =>
        c.per_lod_regressions.length === 0 && c.termination.every((t) => t.known && t.uncensored),
    ),
    comparisons,
    release_quality_proven: false,
  };
}

export function validateOptimizationPilot({
  request,
  seed,
  strategy,
  contract,
  result,
  latest,
  hashes,
  capability = false,
}) {
  if (
    capability &&
    (request.capability_minutes !== 60 || seed !== 101 || strategy !== 'exhaustive')
  )
    throw new Error('invalid capability training contract');
  const expected = {
    version: 9,
    architecture: 4,
    hidden_width: 64,
    seed,
    teacher_strategy: strategy,
    initialize: request.initialization_sha256,
    warmstart: '',
    duration_minutes: capability ? request.capability_minutes : request.learning_minutes,
    finalize_minutes: request.finalize_minutes,
    states: 16,
    updates: 128,
    batch: 512,
    workers: request.workers,
    candidate_batch: request.candidate_batch,
    gpu_memory_mib: request.gpu_memory_mib,
    training_profile: 'coverage',
    mask_only_coverage: true,
    raster: 'vulkan-v1',
    storage: 'packed',
    policy_action_candidates: true,
    policy_rollout_trials: 64,
    episode_seeds: true,
    simplifier_seeds: true,
    quality: false,
    update_backend: 'fp32-compensated-v2',
    ...hashes,
  };
  for (const [key, value] of Object.entries(expected))
    if (contract[key] !== value) throw new Error('paired pilot contract mismatch: ' + key);
  if (
    capability &&
    (result.status !== 'duration_complete' ||
      !Number.isFinite(result.learning_elapsed_ms) ||
      result.learning_elapsed_ms < 60 * 60000)
  )
    throw new Error('capability run did not complete the learning hour');
  if (
    result.complete !== true ||
    latest.complete !== true ||
    !Number.isSafeInteger(result.updates_this_invocation) ||
    result.updates_this_invocation <= 0 ||
    result.failed_conditions_count !== 0 ||
    !latest.checkpoint ||
    result.checkpoint !== latest.checkpoint ||
    result.checkpoint_sha256 !== latest.checkpoint_sha256 ||
    !/^[a-f0-9]{64}$/.test(latest.checkpoint_sha256 ?? '')
  )
    throw new Error('paired pilot incomplete, stale or numerically invalid');
}

export function validateOptimizationCheckpoint({
  latest,
  index,
  verification,
  checkpointSha256,
  modelSha256,
  verificationSha256,
}) {
  if (
    latest.checkpoint_sha256 !== checkpointSha256 ||
    index.complete !== true ||
    index.step !== latest.step ||
    index.checkpoint_sha256 !== checkpointSha256 ||
    index.model_sha256 !== modelSha256 ||
    index.files?.['checkpoint.pt'] !== checkpointSha256 ||
    index.files?.['model.blzn'] !== modelSha256 ||
    index.files?.['verification.json'] !== verificationSha256 ||
    verification.passed !== true
  )
    throw new Error('checkpoint files or verification differ from the published journal');
}
