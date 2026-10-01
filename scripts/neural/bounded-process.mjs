import { spawn } from 'node:child_process';

// Own the complete process group: tools such as Compute Sanitizer launch a
// second process. A signal exit has code=null and must never mean success.
export async function boundedProcess(
  command,
  args,
  { maximum, signal, stdio = 'ignore', env = process.env, grace = 3000 } = {},
) {
  if (!Number.isFinite(maximum) || maximum <= 0 || !Number.isFinite(grace) || grace < 0)
    throw new Error('Invalid process deadline');
  if (signal?.aborted)
    return { code: null, signal: null, timed_out: false, cancelled: true, success: false };
  const child = spawn(command, args, { stdio, env, detached: true });
  let hard,
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
    kill('SIGTERM');
    hard = setTimeout(() => kill('SIGKILL'), grace);
  };
  const abort = () => {
    cancelled = true;
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
    return {
      ...exit,
      timed_out,
      cancelled,
      success: exit.code === 0 && exit.signal === null && !timed_out && !cancelled,
    };
  } finally {
    kill('SIGKILL');
    clearTimeout(timer);
    clearTimeout(hard);
    signal?.removeEventListener('abort', abort);
  }
}
