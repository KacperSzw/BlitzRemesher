// Bounded paired teacher experiment on the rented GPU. No provider credentials.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { read, write } from './artifacts.mjs';
import { boundedProcess } from './bounded-process.mjs';
import { runCoreValidation } from './core-validation-job.mjs';
import { runTeacherProfile } from './teacher-profile.mjs';
import { runNativeDebuggerPreflight } from './native-debugger-preflight.mjs';
import { verifyNvidiaIcdSelection } from './nvidia-icd.mjs';
import { verifyBaselineBuild } from './baseline-overlay.mjs';
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
  debuggerCommand = 'gdb',
  debuggerPreflight = runNativeDebuggerPreflight,
  teardownBinary = 'build/neural/blitz-neural-vulkan-tests',
  icdSelection,
  baselineBuild,
  environment = process.env,
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
  const timeoutDiagnostics = request.timeout_diagnostics
    ? { debuggerCommand, maximum: 5000 }
    : undefined;
  const phase = (name) => {
    report.phase = name;
    write(path.join(directory, '../phase.json'), { phase: name, at: now() });
    persist();
  };
  const nativeExecute = (command, args, options) => {
    if (!report.vulkan_icd) return execute(command, args, options);
    const env = {
      ...environment,
      ...options.env,
      VK_DRIVER_FILES: report.vulkan_icd.selected_path,
    };
    delete env.VK_ICD_FILENAMES;
    return execute(command, args, { ...options, env });
  };
  async function run(command, args, name, end, capture, teardownTrace = true) {
    if (signal?.aborted || now() >= end) throw new Error(name + ': deadline/cancellation');
    const log = directory + '/' + name + '.log',
      fd = fs.openSync(log, 'w');
    try {
      const grace = capture ? capture.maximum + 1000 : 3000;
      const maximum = Math.min(end, deadline) - now() - (capture ? grace : 0);
      if (maximum <= 0) throw new Error(name + ': insufficient deadline for bounded capture');
      const result = await nativeExecute(command, args, {
        maximum,
        grace,
        signal,
        stdio: ['ignore', fd, fd],
        ...(capture
          ? {
              env: {
                ...environment,
                BLITZ_TEARDOWN_TRACE: teardownTrace ? '1' : '0',
                BLITZ_ALLOW_DEBUGGER_ATTACH: '1',
              },
              timeoutDiagnostic: { ...capture, output: directory + '/' + name + '.threads.log' },
            }
          : {}),
      });
      report.phases.push({
        name,
        command,
        args,
        ...(capture ? { teardown_trace: teardownTrace, debugger_attach_requested: true } : {}),
        ...result,
      });
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
    if (request.baseline_overlay !== undefined && !baselineBuild)
      throw new Error('Requested baseline overlay needs verified build provenance');
    if (baselineBuild) {
      report.baseline_build = verifyBaselineBuild({
        evidenceFile: baselineBuild,
        binary: baseline,
        request,
        candidateRoot: process.cwd(),
      });
      persist();
    }
    if (request.vulkan_icd !== undefined && !icdSelection)
      throw new Error('Requested NVIDIA ICD needs verified selection provenance');
    if (icdSelection) {
      const selected = verifyNvidiaIcdSelection(icdSelection, request.vulkan_icd ?? 'vendor');
      if (environment.VK_DRIVER_FILES !== selected.selected_path || environment.VK_ICD_FILENAMES)
        throw new Error('Vulkan environment does not match the verified NVIDIA ICD selection');
      report.vulkan_icd = {
        ...selected,
        VK_DRIVER_FILES: selected.selected_path,
        VK_ICD_FILENAMES: null,
      };
      persist();
    }
    const teacherEnd = Math.min(started + 20 * 60000, deadline),
      learningEnd = Math.min(started + 62 * 60000, deadline);
    const optimized = 'build/neural/blitz-neural-placement-prepare';
    if (timeoutDiagnostics) {
      phase('native-debugger-preflight');
      report.debugger_preflight = await debuggerPreflight({
        binary: optimized,
        directory: directory + '/debugger-preflight',
        debuggerCommand,
        diagnosticMaximum: timeoutDiagnostics.maximum,
        deadline: Math.min(now() + 10000, teacherEnd),
        signal,
        execute: nativeExecute,
        now,
      });
      persist();
      if (!report.debugger_preflight.complete)
        throw new Error('Native debugger preflight failed; timeout stacks are unavailable');
      phase('native-teardown-stress');
      report.teardown_stress = {
        binary: teardownBinary,
        binary_sha256: digest(teardownBinary),
        retirement_mode: 'join',
        fixture_arguments: ['--teardown-join'],
        teardown_trace_by_run: [true, false, false, true],
        requested_runs: 4,
        rounds_per_run: 12,
        requested_rounds: 48,
        completed_runs: 0,
        completed_full_run_rounds: 0,
      };
      persist();
      for (let repeat = 0; repeat < report.teardown_stress.requested_runs; repeat++) {
        await run(
          teardownBinary,
          report.teardown_stress.fixture_arguments,
          'teardown-stress-' + (repeat + 1),
          Math.min(now() + 45000 + timeoutDiagnostics.maximum + 1000, teacherEnd),
          timeoutDiagnostics,
          report.teardown_stress.teardown_trace_by_run[repeat],
        );
        report.teardown_stress.completed_runs++;
        report.teardown_stress.completed_full_run_rounds += report.teardown_stress.rounds_per_run;
        persist();
      }
    }
    phase('coverage-contracts');
    report.contracts = await contracts({
      directory: directory + '/contracts',
      deadline: Math.min(now() + 5 * 60000, teacherEnd),
      signal,
      execute: nativeExecute,
      now,
    });
    persist();
    if (!report.contracts.complete) throw new Error('remote engineering contracts incomplete');
    phase('warm-teacher-comparison');
    report.reuse = await profile({
      baseline,
      optimized,
      directory: directory + '/reuse',
      model,
      plan: 'research/neural/teacher-profile.json',
      deadline: teacherEnd,
      signal,
      execute: nativeExecute,
      now,
      workers: request.workers,
      candidateBatch: request.candidate_batch,
      gpuMemoryMiB: request.gpu_memory_mib,
      maxSeconds: 100,
      timingAuthority: 'isolated_remote',
      timeoutDiagnostics,
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
      execute: nativeExecute,
      now,
      workers: request.workers,
      candidateBatch: request.candidate_batch,
      gpuMemoryMiB: request.gpu_memory_mib,
      maxSeconds: 100,
      timingAuthority: 'isolated_remote',
      comparison: 'strategy',
      timeoutDiagnostics,
    });
    persist();
    if (!report.strategy.complete) throw new Error('warm strategy comparison incomplete');
    const freshStateSpeedup = report.strategy.strategy_fresh_states_speedup;
    report.learning_gate = {
      metric: 'strategy_fresh_states_speedup',
      value: Number.isFinite(freshStateSpeedup) ? freshStateSpeedup : null,
      timing_authority: report.strategy.timing_authority ?? null,
      passed:
        report.strategy.timing_authority === 'isolated_remote' &&
        Number.isFinite(freshStateSpeedup) &&
        freshStateSpeedup > 1,
    };
    if (!report.learning_gate.passed) {
      report.learning_skipped_reason =
        'No finite isolated fresh-state throughput gain above 1; paired learning and quality evaluation were not started.';
      report.strategy_promotable = false;
      phase('teacher-strategy-rejected');
      return report;
    }
    persist();

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
        await run(
          cycle,
          args,
          'seed-' + seed + '-' + strategy,
          now() + 7 * 60000,
          timeoutDiagnostics,
        );
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
    icdSelection: '/workspace/results/nvidia-icd-selection.json',
    baselineBuild: '/workspace/results/baseline-build.json',
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
