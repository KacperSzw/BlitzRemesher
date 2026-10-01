import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { boundedProcess } from './bounded-process.mjs';
import { write } from './artifacts.mjs';
import { coreValidationSteps, validateCoreTestLog } from './core-validation.mjs';

export async function runCoreValidation({
  directory,
  deadline,
  signal,
  execute = boundedProcess,
  now = Date.now,
}) {
  const report = {
    experiment: 'core-validation',
    complete: false,
    training_started: false,
    score: null,
    quality_proven: false,
    deadline,
    phases: [],
  };
  fs.mkdirSync(directory, { recursive: true });
  try {
    if (!Number.isFinite(deadline)) throw new Error('invalid core validation deadline');
    for (const step of coreValidationSteps(directory)) {
      if (signal?.aborted || deadline <= now())
        throw new Error('core validation deadline/cancellation');
      const log = directory + '/' + step.name + '.log',
        fd = fs.openSync(log, 'w');
      try {
        const result = await execute(step.command, step.args, {
          stdio: ['ignore', fd, fd],
          maximum: Math.min(step.maximum, deadline - now()),
          signal,
          env: { ...process.env, ...step.env },
        });
        report.phases.push({ name: step.name, command: step.command, args: step.args, ...result });
        if (
          !result.success ||
          result.code !== 0 ||
          result.signal ||
          result.timed_out ||
          result.cancelled
        )
          throw new Error(step.name + ' failed: ' + JSON.stringify(result));
        validateCoreTestLog(step.name, fs.readFileSync(log, 'utf8'));
      } finally {
        fs.closeSync(fd);
        write(directory + '/report.json', report);
      }
    }
    if (signal?.aborted || deadline <= now())
      throw new Error('core validation deadline/cancellation');
    report.complete = true;
  } catch (error) {
    report.error = String(error);
  } finally {
    report.finished = now();
    write(directory + '/report.json', report);
  }
  return report;
}
async function main() {
  const [setup, latest, minutes] = process.argv.slice(2).map(Number),
    started = Date.now(),
    directory = '/workspace/results/core-validation';
  if (
    ![setup, latest, minutes].every(Number.isFinite) ||
    started >= setup ||
    minutes !== 10 ||
    latest <= started
  )
    throw new Error('invalid core validation deadline');
  const deadline = Math.min(latest, started + minutes * 60000),
    controller = new AbortController();
  for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => controller.abort());
  write('/workspace/results/setup-complete.json', {
    at: started,
    training_minutes: 0,
    test_minutes: minutes,
    training_deadline_ms: deadline,
  });
  const report = await runCoreValidation({ directory, deadline, signal: controller.signal });
  process.exitCode = report.complete ? 0 : 1;
  write('/workspace/results/job.json', {
    code: process.exitCode,
    experiment: 'core-validation',
    training_started: false,
    result: directory + '/report.json',
  });
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
