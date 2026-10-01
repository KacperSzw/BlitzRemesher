// Five-minute calibration, then a fresh sustained experiment. Runs on the Pod only.
import fs from 'node:fs';
import { spawn } from 'node:child_process';
import { read, write } from './artifacts.mjs';
import {
  trainerArguments,
  selectCalibration,
  trainingWindow,
  trainingDeadline as cutoff,
} from './training.mjs';

const setupDeadline = Number(process.argv[2]),
  latestTrainingDeadline = Number(process.argv[3]),
  trainingMinutes = Number(process.argv[4]);
if (
  !Number.isFinite(setupDeadline) ||
  !Number.isFinite(latestTrainingDeadline) ||
  !Number.isFinite(trainingMinutes) ||
  trainingMinutes <= 0
)
  throw new Error('Missing absolute deadlines or training duration');
const results = '/workspace/results',
  dataset = '/workspace/dataset',
  run = results + '/experiment';
const base = {
  core: 4096,
  bootstrap_steps: 100,
  health_steps: 8192,
  checkpoint_every: 1024,
  stage_steps: 25000,
  max_steps: 100000,
  max_stalled_pilots: 2,
  train_until_deadline: true,
  hours: 2,
  gpu_memory_mib: 12288,
};
let active,
  cancelled = false;
for (const signal of ['SIGINT', 'SIGTERM'])
  process.on(signal, () => {
    cancelled = true;
    if (active) process.kill(-active.pid, 'SIGTERM');
  });

async function execute(command, args, log, deadline) {
  if (cancelled || Date.now() >= deadline)
    throw new Error('Remote phase deadline or cancellation reached');
  const fd = fs.openSync(log, 'a');
  active = spawn(command, args, { stdio: ['ignore', fd, fd], detached: true });
  let killed = false,
    hard,
    reason;
  const stop = () => {
    if (killed || !active) return;
    killed = true;
    process.kill(-active.pid, 'SIGTERM');
    hard = setTimeout(() => {
      if (active) process.kill(-active.pid, 'SIGKILL');
    }, 10000);
  };
  const timer = setTimeout(
    () => {
      reason = 'deadline';
      stop();
    },
    Math.max(1, deadline - Date.now()),
  );
  // Sum process RSS for this process group; stop at the protocol's 24 GiB host budget.
  const memory = setInterval(() => {
    let kib = 0;
    for (const pid of fs.readdirSync('/proc').filter((p) => /^\d+$/.test(p))) {
      try {
        const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8'),
          fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
        if (Number(fields[2]) !== active?.pid) continue;
        const match = fs.readFileSync(`/proc/${pid}/status`, 'utf8').match(/^VmRSS:\s+(\d+)/m);
        kib += Number(match?.[1] ?? 0);
      } catch {} // Processes can exit between /proc reads.
    }
    if (kib > 24 * 1024 * 1024) {
      reason = 'host_memory_budget';
      stop();
    }
  }, 1000);
  try {
    return await new Promise((resolve, reject) => {
      active.once('error', reject);
      active.once('close', (code) => resolve({ code, stopped: killed, reason }));
    });
  } finally {
    clearInterval(memory);
    clearTimeout(timer);
    clearTimeout(hard);
    active = undefined;
    fs.closeSync(fd);
  }
}

const calibrationDeadline = Math.min(Date.now() + 5 * 60000, setupDeadline),
  trials = [];
for (const [batch, workers] of [
  [64, 4],
  [64, 8],
  [128, 8],
]) {
  const dir = `${results}/calibration/b${batch}-w${workers}`,
    config = { ...base, batch, workers };
  fs.mkdirSync(dir, { recursive: true });
  const start = Date.now(),
    samples = [],
    telemetry = fs.createWriteStream(dir + '/gpu.jsonl');
  let pending = '',
    monitorError;
  const monitor = spawn('nvidia-smi', [
    '--query-gpu=utilization.gpu,power.draw,memory.used',
    '--format=csv,noheader,nounits',
    '-l',
    '1',
  ]);
  monitor.on('error', (error) => {
    monitorError = String(error);
  });
  monitor.stdout.on('data', (chunk) => {
    pending += chunk.toString();
    let at;
    while ((at = pending.indexOf('\n')) >= 0) {
      const values = pending.slice(0, at).split(',').map(Number);
      pending = pending.slice(at + 1);
      if (values.length !== 3 || values.some((v) => !Number.isFinite(v))) continue;
      const [gpu, power_w, memory_mib] = values,
        row = { at: new Date().toISOString(), phase: 'training', gpu, power_w, memory_mib };
      samples.push(row);
      telemetry.write(JSON.stringify(row) + '\n');
    }
  });
  try {
    const outcome = await execute(
      'build/neural/blitz-neural-train',
      trainerArguments(dataset, dir, null, config, 1000000, 1.25),
      dir + '/train.log',
      Math.min(start + 90000, calibrationDeadline),
    );
    if (monitorError) throw new Error(monitorError);
    const health = read(dir + '/latest.json');
    const metrics = fs
      .readFileSync(dir + '/metrics.jsonl', 'utf8')
      .trim()
      .split('\n')
      .map(JSON.parse)
      .slice(16);
    const seconds = metrics.reduce((sum, r) => sum + r.seconds, 0),
      vertices = metrics.reduce((sum, r) => sum + r.core_vertices, 0);
    trials.push({
      batch,
      workers,
      ...outcome,
      health,
      gpu: trainingWindow(samples),
      measured_steps: metrics.length,
      vertices_per_second: vertices / seconds,
      elapsed_seconds: (Date.now() - start) / 1000,
    });
  } catch (error) {
    trials.push({
      batch,
      workers,
      error: String(error),
      gpu: trainingWindow(samples),
      elapsed_seconds: (Date.now() - start) / 1000,
    });
  } finally {
    monitor.kill('SIGTERM');
    telemetry.end();
  }
  write(results + '/calibration.json', { complete: false, trials });
}
const best = selectCalibration(
  trials.filter((t) => (t.code === 0 || t.code === 2) && t.reason !== 'host_memory_budget'),
);
write(results + '/calibration.json', {
  complete: true,
  trials,
  selected: { batch: best.batch, workers: best.workers },
});
if (Date.now() >= setupDeadline) throw new Error('Setup exceeded 30-minute allowance');
fs.mkdirSync(run, { recursive: true });
if (fs.existsSync(run + '/training'))
  throw new Error('Fresh cloud experiment directory already has training state');
write(run + '/training.json', { ...base, batch: best.batch, workers: best.workers });
const started = Date.now(),
  trainingDeadline = cutoff(started, latestTrainingDeadline, trainingMinutes);
write(results + '/setup-complete.json', {
  at: started,
  training_minutes: trainingMinutes,
  training_deadline_ms: trainingDeadline,
});
const outcome = await execute(
  process.execPath,
  ['scripts/neural/train.mjs', run, dataset, '--from-scratch', String(trainingDeadline)],
  results + '/experiment.log',
  trainingDeadline + 15000,
);
write(results + '/job.json', outcome);
process.exitCode = outcome.code === 0 ? 0 : 1;
