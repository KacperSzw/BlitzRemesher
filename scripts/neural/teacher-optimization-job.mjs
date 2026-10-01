// Bounded paired teacher experiment on the rented GPU. No provider credentials.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { read, write } from './artifacts.mjs';
import { boundedProcess } from './bounded-process.mjs';
import { runCoreValidation } from './core-validation-job.mjs';
import { runTeacherProfile } from './teacher-profile.mjs';
import {
  validateOptimizationRequest,
  optimizationCycleArguments,
  teacherStrategyGate,
  validateOptimizationPilot,
  validateOptimizationCheckpoint,
} from './teacher-optimization.mjs';

const digest = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');

export async function runTeacherOptimization({
  directory,
  request,
  model,
  deadline,
  signal,
  execute = boundedProcess,
  now = Date.now,
  profile = runTeacherProfile,
  contracts = runCoreValidation,
  baseline = '/workspace/baseline/build/neural/blitz-neural-placement-prepare',
}) {
  validateOptimizationRequest(request);
  const started = now(),
    report = {
      experiment: 'teacher-optimization',
      complete: false,
      final_training: false,
      training_started: false,
      score: null,
      quality_proven: false,
      started,
      deadline,
      request,
      phases: [],
      pilots: [],
      quality: [],
    };
  if (fs.existsSync(directory)) throw new Error('choose a fresh teacher optimization directory');
  fs.mkdirSync(directory, { recursive: true });
  const persist = () => write(directory + '/report.json', report);
  const phase = (name) => {
    report.phase = name;
    write(path.join(directory, '../phase.json'), { phase: name, at: now() });
    persist();
  };
  async function run(command, args, name, end) {
    if (signal?.aborted || now() >= end) throw new Error(name + ': deadline/cancellation');
    const log = directory + '/' + name + '.log',
      fd = fs.openSync(log, 'w');
    try {
      const result = await execute(command, args, {
        maximum: Math.min(end, deadline) - now(),
        signal,
        stdio: ['ignore', fd, fd],
      });
      report.phases.push({ name, command, args, ...result });
      if (
        !result.success ||
        result.code !== 0 ||
        result.signal ||
        result.timed_out ||
        result.cancelled
      )
        throw new Error(name + ' failed: ' + JSON.stringify(result));
    } finally {
      fs.closeSync(fd);
      persist();
    }
  }
  try {
    if (
      !Number.isFinite(deadline) ||
      deadline <= started ||
      digest(model) !== request.initialization_sha256
    )
      throw new Error('invalid experiment deadline or initializer checksum');
    const teacherEnd = Math.min(started + 20 * 60000, deadline),
      learningEnd = Math.min(started + 62 * 60000, deadline);
    phase('coverage-contracts');
    report.contracts = await contracts({
      directory: directory + '/contracts',
      deadline: Math.min(started + 5 * 60000, teacherEnd),
      signal,
      execute,
      now,
    });
    persist();
    if (!report.contracts.complete) throw new Error('remote engineering contracts incomplete');
    const optimized = 'build/neural/blitz-neural-placement-prepare';
    phase('warm-teacher-comparison');
    report.reuse = await profile({
      baseline,
      optimized,
      directory: directory + '/reuse',
      model,
      plan: 'research/neural/teacher-profile.json',
      deadline: teacherEnd,
      signal,
      execute,
      now,
      workers: request.workers,
      candidateBatch: request.candidate_batch,
      gpuMemoryMiB: request.gpu_memory_mib,
      maxSeconds: 100,
      timingAuthority: 'isolated_remote',
    });
    persist();
    if (!report.reuse.complete) throw new Error('warm reuse comparison incomplete or unequal');
    report.strategy = await profile({
      baseline: optimized,
      optimized,
      directory: directory + '/strategy',
      model,
      plan: 'research/neural/teacher-profile.json',
      deadline: teacherEnd,
      signal,
      execute,
      now,
      workers: request.workers,
      candidateBatch: request.candidate_batch,
      gpuMemoryMiB: request.gpu_memory_mib,
      maxSeconds: 100,
      timingAuthority: 'isolated_remote',
      comparison: 'strategy',
    });
    persist();
    if (!report.strategy.complete) throw new Error('warm strategy comparison incomplete');

    const cycle = 'build/neural/blitz-neural-cycle';
    for (const [i, seed] of request.seeds.entries()) {
      if (learningEnd - now() < 14 * 60000)
        throw new Error('another complete paired pilot no longer fits the learning budget');
      // Alternate which strategy runs first while keeping each pair adjacent.
      const strategies =
        i % 2 ? ['coverage-core-first', 'exhaustive'] : ['exhaustive', 'coverage-core-first'];
      for (const strategy of strategies) {
        if (learningEnd - now() < 7 * 60000)
          throw new Error('another complete paired pilot no longer fits the learning budget');
        const runDirectory = directory + '/seed-' + seed + '-' + strategy;
        const args = optimizationCycleArguments(request, {
          run: runDirectory,
          seed,
          strategy,
          model,
          curriculum: 'research/neural/teacher-optimization-curriculum.json',
        });
        report.training_started = true;
        phase('paired-learning-pilots');
        await run(cycle, args, 'seed-' + seed + '-' + strategy, now() + 7 * 60000);
        const result = read(runDirectory + '/report.json'),
          latest = read(runDirectory + '/latest.json'),
          contract = read(runDirectory + '/contract.json');
        validateOptimizationPilot({
          request,
          seed,
          strategy,
          contract,
          result,
          latest,
          hashes: {
            binary_sha256: digest(cycle),
            curriculum_sha256: digest('research/neural/teacher-optimization-curriculum.json'),
            corpus_sha256: digest('research/neural/corpus-v2/corpus.json'),
            training_selection_sha256: digest('research/neural/corpus-v2/training.json'),
          },
        });
        const checkpoint = path.resolve(runDirectory, latest.checkpoint);
        if (!checkpoint.startsWith(path.resolve(runDirectory) + '/'))
          throw new Error('invalid pilot checkpoint path');
        const pilot = {
          seed,
          strategy,
          directory: runDirectory,
          report: result,
          contract,
          model: checkpoint + '/model.blzn',
          model_sha256: digest(checkpoint + '/model.blzn'),
          checkpoint_sha256: digest(checkpoint + '/checkpoint.pt'),
        };
        const index = read(checkpoint + '/index.json'),
          verification = read(checkpoint + '/verification.json');
        validateOptimizationCheckpoint({
          latest,
          index,
          verification,
          checkpointSha256: pilot.checkpoint_sha256,
          modelSha256: pilot.model_sha256,
          verificationSha256: digest(checkpoint + '/verification.json'),
        });
        pilot.verification = verification;
        report.pilots.push(pilot);
        persist();
      }
    }

    const qualityEnd = Math.min(now() + 50 * 60000, deadline),
      settings = read('research/neural/teacher-optimization-quality.json');
    phase('full-lod-quality-comparison');
    if (
      settings.action_trials !== request.action_trials ||
      settings.action_batch !== request.action_batch ||
      settings.gpu_memory_mib !== request.gpu_memory_mib ||
      settings.levels !== 8
    )
      throw new Error('quality settings differ from the frozen experiment');
    const manifest = 'research/neural/prepared-pilot/selection.json',
      models = new Map();
    for (const pilot of report.pilots) {
      const name = 'quality-' + pilot.seed + '-' + pilot.strategy,
        output = directory + '/' + name + '.json';
      await run(
        'build/neural/blitz-neural-diagnostics',
        [
          '--audit-model',
          manifest,
          pilot.model,
          'research/neural/teacher-optimization-quality.json',
          output,
        ],
        name,
        qualityEnd,
      );
      const audit = read(output);
      if (audit.model_sha256 !== pilot.model_sha256 || digest(pilot.model) !== pilot.model_sha256)
        throw new Error('quality comparison model checksum changed');
      report.quality.push({ seed: pilot.seed, strategy: pilot.strategy, output, audit });
      models.set(pilot.seed + '/' + pilot.strategy, audit);
      persist();
    }
    report.strategy_gate = teacherStrategyGate(
      request.seeds.map((seed) => ({
        seed,
        exhaustive: models.get(seed + '/exhaustive'),
        candidate: models.get(seed + '/coverage-core-first'),
      })),
      read(manifest).assets.length,
    );
    report.curriculum_coverage_complete = report.pilots.every(
      (p) => p.report.curriculum_coverage_complete,
    );
    report.strategy_promotable =
      report.strategy_gate.passed &&
      report.curriculum_coverage_complete &&
      report.strategy.timing_authority === 'isolated_remote' &&
      report.strategy.strategy_fresh_states_speedup > 1;
    report.complete = true;
  } catch (error) {
    report.error = String(error);
    report.strategy_promotable = false;
  } finally {
    report.finished = now();
    persist();
  }
  return report;
}

async function main() {
  const [setup, latest, minutes] = process.argv.slice(2).map(Number),
    started = Date.now();
  if (
    ![setup, latest, minutes].every(Number.isFinite) ||
    started >= setup ||
    minutes !== 112 ||
    latest <= started
  )
    throw new Error('invalid teacher optimization deadline');
  const deadline = Math.min(latest, started + minutes * 60000),
    cancellation = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => cancellation.abort());
  write('/workspace/results/setup-complete.json', {
    at: started,
    training_minutes: 30,
    experiment_minutes: minutes,
    training_deadline_ms: deadline,
  });
  const report = await runTeacherOptimization({
    directory: '/workspace/results/teacher-optimization',
    request: read('/workspace/optimization/request.json'),
    model: '/workspace/optimization/model.blzn',
    deadline,
    signal: cancellation.signal,
  });
  process.exitCode = report.complete ? 0 : 1;
  write('/workspace/results/job.json', {
    code: process.exitCode,
    experiment: 'teacher-optimization',
    training_started: report.training_started,
    final_training: false,
    result: '/workspace/results/teacher-optimization/report.json',
  });
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
