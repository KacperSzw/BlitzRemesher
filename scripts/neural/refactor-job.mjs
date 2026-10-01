// One bounded rental: remote validation, report, then the full learning cycle.
// No provider credential is present in this process or in the rented container.
import fs from 'node:fs';
import { spawn } from 'node:child_process';
import { read, write } from './artifacts.mjs';
import { validateCoverage, coverageArchitecture } from './coverage-validation.mjs';
import { persistentLearningCycle, validatePreparedLearning } from './resident-cycle.mjs';
const [setup, latest, minutes] = process.argv.slice(2).map(Number),
  started = Date.now(),
  results = '/workspace/results',
  root = results + '/gpu-refactor';
if (![setup, latest, minutes].every(Number.isFinite) || started >= setup || minutes !== 150)
  throw new Error('Invalid refactor rental deadline');
const deadline = Math.min(latest, started + minutes * 60000);
if (deadline - started < 134 * 60000 + 15000)
  throw new Error('Full learning plus finalization no longer fit');
fs.mkdirSync(root, { recursive: true });
write(results + '/setup-complete.json', {
  at: started,
  training_minutes: minutes,
  training_deadline_ms: deadline,
});
let phase = 'validation',
  active,
  cancelled = false,
  activeDeadline = Math.min(started + 16 * 60000, deadline - 134 * 60000 - 15000);
const setPhase = (value) => {
  phase = value;
  write(results + '/phase.json', { phase, at: Date.now() });
};
setPhase(phase);
const terminate = (child) => {
  if (child && !child.exitCode)
    try {
      process.kill(-child.pid, 'SIGTERM');
    } catch (e) {
      if (e.code !== 'ESRCH') throw e;
    }
};
for (const signal of ['SIGTERM', 'SIGINT'])
  process.on(signal, () => {
    cancelled = true;
    terminate(active);
  });
const monitor = spawn('nvidia-smi', [
  '--query-gpu=utilization.gpu,power.draw,memory.used',
  '--format=csv,noheader,nounits',
  '-l',
  '1',
]);
const telemetry = fs.createWriteStream(root + '/gpu.jsonl');
let pending = '',
  monitorError;
monitor.on('error', (e) => {
  monitorError = String(e);
});
monitor.stdout.on('data', (chunk) => {
  pending += chunk;
  let at;
  while ((at = pending.indexOf('\n')) >= 0) {
    const v = pending.slice(0, at).split(',').map(Number);
    pending = pending.slice(at + 1);
    if (v.length === 3 && v.every(Number.isFinite))
      telemetry.write(
        JSON.stringify({ at: Date.now(), phase, gpu: v[0], power_w: v[1], memory_mib: v[2] }) +
          '\n',
      );
  }
});
async function execute(name, args, log, maximumMinutes, environment = {}) {
  if (cancelled || Date.now() + 5000 >= activeDeadline)
    throw new Error('Refactor deadline/cancellation');
  const executable =
    name === 'compute-sanitizer' ? '/usr/local/cuda/bin/compute-sanitizer' : 'build/neural/' + name;
  const fd = fs.openSync(log, 'a'),
    end = Math.min(activeDeadline - 5000, Date.now() + maximumMinutes * 60000),
    child = spawn(executable, args, {
      stdio: ['ignore', fd, fd],
      detached: true,
      env: { ...process.env, ...environment },
    });
  active = child;
  let hard,
    timedOut = false,
    memoryExceeded = false;
  const stop = () => {
    terminate(child);
    hard ??= setTimeout(() => {
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch (e) {
        if (e.code !== 'ESRCH') throw e;
      }
    }, 5000);
  };
  const timer = setTimeout(
    () => {
      timedOut = true;
      stop();
    },
    Math.max(1, end - Date.now()),
  );
  const memory = setInterval(() => {
    let kib = 0;
    for (const pid of fs.readdirSync('/proc').filter((p) => /^\d+$/.test(p))) {
      try {
        const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8'),
          fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
        if (Number(fields[2]) !== child.pid) continue;
        kib += Number(
          fs.readFileSync(`/proc/${pid}/status`, 'utf8').match(/^VmRSS:\s+(\d+)/m)?.[1] ?? 0,
        );
      } catch {}
    }
    if (kib > 24 * 1024 * 1024) {
      memoryExceeded = true;
      stop();
    }
  }, 1000);
  try {
    const code = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', resolve);
    });
    if (cancelled || memoryExceeded || timedOut || ![0, 2].includes(code))
      throw new Error(
        `${name} failed (exit=${code}, cancelled=${cancelled}, deadline=${timedOut}, memory=${memoryExceeded})`,
      );
    return code;
  } finally {
    clearTimeout(timer);
    clearTimeout(hard);
    clearInterval(memory);
    active = undefined;
    fs.closeSync(fd);
  }
}
const result = {
  started,
  deadline,
  complete: false,
  validation_passed: false,
  quality_proven: false,
};
try {
  await execute(
    'compute-sanitizer',
    ['--tool', 'memcheck', '--error-exitcode', '1', 'build/neural/blitz-neural-action-gpu-tests'],
    root + '/memcheck.log',
    2,
  );
  const ctx = { root, execute, phase: setPhase, latest: deadline },
    validation = await validateCoverage(ctx);
  result.preflight = await validatePreparedLearning(ctx, validation);
  result.validation_passed = true;
  result.architecture_written_at = Date.now();
  fs.writeFileSync(root + '/ARCHITECTURE.md', coverageArchitecture(validation));

  activeDeadline = deadline;
  result.cycle = await persistentLearningCycle(ctx, validation);
  result.complete = result.cycle.complete;
  if (!result.complete) throw new Error('Learning cycle did not meet completion contracts');
  setPhase('final-validation-comparison');
  const finalAudit = root + '/final-model-audit.json';
  await execute(
    'blitz-neural-diagnostics',
    [
      '--audit-model',
      read('research/neural/next-training.json').validation_selection,
      root + '/resident-cycle/' + result.cycle.checkpoint + '/model.blzn',
      result.preflight.settings,
      finalAudit,
    ],
    root + '/final-model-audit.log',
    4,
  );
  result.final_audit = read(finalAudit);
  result.complete = result.complete && result.final_audit.complete;
  if (!result.complete) throw new Error('Final model comparison incomplete');
  if (monitorError) throw new Error(monitorError);
} catch (error) {
  result.complete = false;
  result.error = String(error);
  process.exitCode = 1;
} finally {
  result.finished = Date.now();
  if (fs.existsSync(root + '/cycle.json')) result.cycle = read(root + '/cycle.json');
  write(root + '/result.json', result);
  write(results + '/job.json', {
    code: process.exitCode ?? 0,
    experiment: 'gpu-refactor',
    result: root + '/result.json',
  });
  setPhase('finished');
  monitor.kill('SIGTERM');
  telemetry.end();
}
