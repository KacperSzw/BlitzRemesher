import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';

export function validateStackDiagnostic({ debuggerCommand, maximum, output }) {
  if (
    typeof debuggerCommand !== 'string' ||
    !debuggerCommand ||
    !Number.isInteger(maximum) ||
    maximum < 1 ||
    maximum > 30000 ||
    typeof output !== 'string' ||
    !output
  )
    throw new Error('Invalid native stack diagnostic');
}

// The debugger is a separate process group. Its deadline includes forced
// cleanup, and returning never depends on a stuck debugger delivering close.
export async function captureNativeStacks(pid, options) {
  validateStackDiagnostic(options);
  if (!Number.isSafeInteger(pid) || pid <= 0) throw new Error('Invalid diagnostic PID');
  const { debuggerCommand, maximum, output, signal } = options;
  const result = {
    attempted: false,
    captured: false,
    pid,
    command: debuggerCommand,
    output,
    maximum_ms: maximum,
    timed_out: false,
    cancelled: false,
    code: null,
    signal: null,
  };
  if (signal?.aborted) return { ...result, cancelled: true };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const fd = fs.openSync(output, 'wx');
  let child,
    timer,
    settle,
    finished = false;
  const kill = () => {
    if (!child?.pid) return;
    try {
      process.kill(-child.pid, 'SIGKILL');
    } catch (error) {
      if (error.code !== 'ESRCH') result.cleanup_error = String(error);
    }
  };
  const abort = () => {
    result.cancelled = true;
    kill();
    settle?.();
  };
  try {
    const done = new Promise((resolve) => {
      settle = resolve;
    });
    const args = [
      '--nx',
      '--nh',
      '--batch',
      '--quiet',
      '-iex',
      'set auto-load off',
      '-iex',
      'set debuginfod enabled off',
      '-ex',
      'set pagination off',
      '-ex',
      'set print frame-arguments none',
      '-ex',
      'set print thread-events off',
      '-ex',
      `attach ${pid}`,
      '-ex',
      'thread apply all bt 32',
      '-ex',
      'detach',
      '-ex',
      'quit',
    ];
    child = spawn(debuggerCommand, args, {
      detached: true,
      stdio: ['ignore', fd, fd],
      env: { ...process.env, DEBUGINFOD_URLS: '', LC_ALL: 'C' },
    });
    result.attempted = true;
    child.once('error', (error) => {
      if (!finished) {
        result.error = String(error);
        settle();
      }
    });
    child.once('close', (code, signal) => {
      if (!finished) {
        result.code = code;
        result.signal = signal;
        settle();
      }
    });
    signal?.addEventListener('abort', abort, { once: true });
    timer = setTimeout(() => {
      result.timed_out = true;
      kill();
      settle();
    }, maximum);
    if (signal?.aborted) abort();
    await done;
  } catch (error) {
    result.error = String(error);
  } finally {
    finished = true;
    kill();
    child?.unref();
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
    fs.closeSync(fd);
  }
  const bytes = fs.readFileSync(output),
    text = bytes.toString('utf8');
  result.output_sha256 = createHash('sha256').update(bytes).digest('hex');
  result.output_bytes = bytes.length;
  // Ubuntu GDB 15.1 can clobber errno while printing its Yama denial advice,
  // leaving "Inappropriate ioctl for device" in the following ptrace error.
  // Recognize that specific permission warning, not arbitrary attach failures.
  result.ptrace_denied =
    /ptrace: (Operation not permitted|Permission denied)|Could not attach.*[Pp]ermission/.test(
      text,
    ) ||
    /Could not attach to process\.\s+If your uid matches the uid of the target\s+process, check the setting of \/proc\/sys\/kernel\/yama\/ptrace_scope\b/.test(
      text,
    );
  result.threads = (text.match(/^Thread\s+\d+\b/gm) ?? []).length;
  result.frames = (text.match(/^#\d+\s/gm) ?? []).length;
  result.captured =
    result.code === 0 &&
    !result.signal &&
    !result.timed_out &&
    !result.cancelled &&
    !result.error &&
    !result.ptrace_denied &&
    result.threads > 0 &&
    result.frames > 0;
  return result;
}
