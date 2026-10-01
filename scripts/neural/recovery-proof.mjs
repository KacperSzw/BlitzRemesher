// Interrupt after a durable mid-wave checkpoint, resume, and compare the
// complete learned policy and ordered labels with an uninterrupted execution.
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { read, write } from './artifacts.mjs';
const [directory] = process.argv.slice(2),
  root = path.resolve(directory ?? '');
if (!directory || fs.existsSync(root)) throw new Error('recovery-proof.mjs NEW_OUTPUT');
fs.mkdirSync(root, { recursive: true });
const report = { complete: false, score: null, rows: [] },
  binary = process.env.BLITZ_CYCLE_BINARY ?? 'build/neural/blitz-neural-cycle';
write(root + '/curriculum.json', {
  version: 1,
  conditions: Array.from({ length: 6 }, (_, i) => ({
    asset: i % 2 ? 'ph_sweet_potato' : 'ph_painted_wooden_bench',
    pixels: i % 2 ? 64 : 32,
    previous_steps: 0,
  })),
});
const options = [
  '--curriculum',
  root + '/curriculum.json',
  '--states',
  '2',
  '--passes',
  '4',
  '--updates',
  '4096',
  '--batch',
  '128',
  '--workers',
  '2',
  '--candidate-batch',
  '4',
  '--episode-seeds',
  'on',
  '--checkpoint-seconds',
  '1',
  '--training-profile',
  'coverage',
  '--quality',
  'off',
  '--update-backend',
  'fused',
  '--gpu-memory-mib',
  process.env.BLITZ_VALIDATION_MEMORY_MIB ?? '1024',
  '--minutes',
  '3',
];
async function run(name, interrupt) {
  const out = root + '/' + name,
    log = root + '/' + name + (interrupt ? '-interrupted' : '-complete') + '.log',
    fd = fs.openSync(log, 'w'),
    start = Date.now();
  const child = spawn(binary, [out, ...options], { stdio: ['ignore', fd, fd] });
  let signalled = false;
  const poll = setInterval(() => {
    if (!interrupt || signalled || !fs.existsSync(out + '/latest.json')) return;
    const journal = read(out + '/latest.json');
    if (journal.active_training && journal.pending_wave) {
      report.interruption = {
        condition: journal.next_condition,
        step:
          read(out + '/' + journal.checkpoint + '/verification.json').step ?? journal.checkpoint,
      };
      child.kill('SIGTERM');
      signalled = true;
    }
  }, 25);
  let hard;
  const timer = setTimeout(() => {
    child.kill('SIGTERM');
    hard = setTimeout(() => child.kill('SIGKILL'), 3000);
  }, 185000);
  try {
    const code = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', resolve);
    });
    report.rows.push({ name, interrupt, code, signalled, seconds: (Date.now() - start) / 1000 });
    write(root + '/report.json', report);
    if (interrupt ? !signalled : code !== 0)
      throw new Error('recovery execution failed: ' + name + ' ' + code);
  } finally {
    clearInterval(poll);
    clearTimeout(timer);
    clearTimeout(hard);
    fs.closeSync(fd);
  }
}
try {
  await run('resumed', true);
  await run('resumed', false);
  await run('continuous', false);
  const a = read(root + '/resumed/report.json'),
    b = read(root + '/continuous/report.json');
  if (!a.complete || !b.complete || a.step !== b.step || a.fresh_states !== b.fresh_states)
    throw new Error('recovery work totals differ');
  const policy = (name) =>
    fs.readFileSync(
      root + '/' + name + '/' + read(root + '/' + name + '/report.json').checkpoint + '/model.blzn',
    );
  if (!policy('resumed').equals(policy('continuous'))) throw new Error('recovered policy differs');
  const labels = (name, r) =>
    r.all_datasets.map(
      (shard) => read(root + '/' + name + '/data/' + shard + '/index.json').sha256,
    );
  if (JSON.stringify(labels('resumed', a)) !== JSON.stringify(labels('continuous', b)))
    throw new Error('recovered label order differs');
  report.complete = true;
  report.step = a.step;
  report.fresh_states = a.fresh_states;
  report.policy_identical = true;
  report.labels_identical = true;
} catch (error) {
  report.error = String(error);
  process.exitCode = 1;
} finally {
  write(root + '/report.json', report);
  console.log(JSON.stringify(report));
}
