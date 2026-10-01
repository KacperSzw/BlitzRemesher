import { spawn } from 'node:child_process';
import { captureNativeStacks, validateStackDiagnostic } from './native-stack.mjs';

// Own the complete process group: tools such as Compute Sanitizer launch a
// second process. A signal exit has code=null and must never mean success.
export async function boundedProcess(
  command,
  args,
  { maximum, signal, stdio = 'ignore', env = process.env, grace = 3000, timeoutDiagnostic } = {},
) {
  if (!Number.isFinite(maximum) || maximum <= 0 || !Number.isFinite(grace) || grace < 0)
    throw new Error('Invalid process deadline');
  if (timeoutDiagnostic) {
    validateStackDiagnostic(timeoutDiagnostic);
    if (grace < timeoutDiagnostic.maximum + 500)
      throw new Error('Process grace must reserve bounded debugger capture and termination');
  }
  if (signal?.aborted)
    return { code: null, signal: null, timed_out: false, cancelled: true, success: false };
  const child = spawn(command, args, { stdio, env, detached: true });
  const diagnosticCancellation = new AbortController();
  let hard,
    diagnostic,
    timed_out = false,
    cancelled = false,
    terminated = false;
  const kill = (value) => {
    if (child.pid)
      try {
        process.kill(-child.pid, value);
      } catch (error) {
        if (error.code !== 'ESRCH') throw error;
      }
  };
  const stop = () => {
    if (terminated) return;
    terminated = true;
    hard = setTimeout(() => kill('SIGKILL'), grace);
    if (timed_out && !cancelled && timeoutDiagnostic) {
      diagnostic = captureNativeStacks(child.pid, {
        ...timeoutDiagnostic,
        signal: diagnosticCancellation.signal,
      })
        .catch((error) => ({ attempted: true, captured: false, error: String(error) }))
        .finally(() => kill('SIGTERM'));
    } else kill('SIGTERM');
  };
  const abort = () => {
    cancelled = true;
    diagnosticCancellation.abort();
    if (terminated) kill('SIGTERM');
    stop();
  };
  signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => {
    timed_out = true;
    stop();
  }, maximum);
  try {
    const exit = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (code, signal) => resolve({ code, signal }));
    });
    diagnosticCancellation.abort();
    const diagnosticResult = diagnostic ? await diagnostic : undefined;
    return {
      ...exit,
      timed_out,
      cancelled,
      success: exit.code === 0 && exit.signal === null && !timed_out && !cancelled,
      ...(diagnosticResult ? { diagnostic: diagnosticResult } : {}),
    };
  } finally {
    kill('SIGKILL');
    diagnosticCancellation.abort();
    clearTimeout(timer);
    clearTimeout(hard);
    signal?.removeEventListener('abort', abort);
  }
}
