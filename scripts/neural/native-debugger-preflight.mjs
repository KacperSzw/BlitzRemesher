import fs from 'node:fs';
import path from 'node:path';
import { boundedProcess } from './bounded-process.mjs';
import { write } from './artifacts.mjs';

// Exercise the deployed executable and debugger under the actual ptrace and
// seccomp policy before relying on timeout stack capture. This starts no GPU.
export async function runNativeDebuggerPreflight({
  binary,
  directory,
  deadline,
  signal,
  debuggerCommand = 'gdb',
  diagnosticMaximum = 5000,
  execute = boundedProcess,
  now = Date.now,
}) {
  if (fs.existsSync(directory)) throw new Error('Choose a fresh debugger preflight directory');
  fs.mkdirSync(directory, { recursive: true });
  const report = {
    complete: false,
    training_started: false,
    gpu_started: false,
    binary,
    debugger_command: debuggerCommand,
    started: now(),
    deadline,
  };
  const log = path.join(directory, 'probe.log'),
    fd = fs.openSync(log, 'wx');
  try {
    const grace = diagnosticMaximum + 1000;
    if (signal?.aborted || !Number.isFinite(deadline) || deadline - now() < grace + 1000)
      throw new Error('Insufficient debugger preflight deadline');
    report.process = await execute(binary, ['--debugger-probe'], {
      maximum: 1000,
      grace,
      signal,
      stdio: ['ignore', fd, fd],
      env: { ...process.env, BLITZ_ALLOW_DEBUGGER_ATTACH: '1' },
      timeoutDiagnostic: {
        debuggerCommand,
        maximum: diagnosticMaximum,
        output: path.join(directory, 'threads.log'),
      },
    });
    const text = fs.readFileSync(log, 'utf8');
    report.target_ready = text.includes('debugger-probe ready');
    report.attach_permission_enabled = text.includes(
      'debugger_attach requested=true permitted=true',
    );
    report.complete =
      report.process.timed_out === true &&
      !report.process.cancelled &&
      report.process.code === 0 &&
      report.process.signal === null &&
      report.target_ready &&
      report.attach_permission_enabled &&
      report.process.diagnostic?.captured === true &&
      now() < deadline &&
      !signal?.aborted;
    if (!report.complete)
      throw new Error('Native debugger preflight did not capture permitted native thread stacks');
  } catch (error) {
    report.error = String(error);
  } finally {
    fs.closeSync(fd);
    report.finished = now();
    write(path.join(directory, 'report.json'), report);
  }
  return report;
}
