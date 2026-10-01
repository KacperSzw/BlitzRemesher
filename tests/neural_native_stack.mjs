import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { boundedProcess } from '../scripts/neural/bounded-process.mjs';
import { captureNativeStacks } from '../scripts/neural/native-stack.mjs';
import { runNativeDebuggerPreflight } from '../scripts/neural/native-debugger-preflight.mjs';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-native-stack-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}
function debuggerScript(root, source) {
  const file = root + '/debugger';
  fs.writeFileSync(file, `#!${process.execPath}\n${source}\n`, { mode: 0o755 });
  return file;
}
function live(pid) {
  try {
    return fs.readFileSync(`/proc/${pid}/stat`, 'utf8').split(') ')[1][0] !== 'Z';
  } catch (error) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}
async function assertStopped(pids) {
  for (let i = 0; i < 40 && pids.some(live); ++i) await delay(25);
  assert.deepEqual(pids.filter(live), [], 'diagnostic must not leave live descendants');
}

test('normal bounded processes never start the optional debugger', async (t) => {
  const root = fixture(t),
    result = await boundedProcess(process.execPath, ['-e', 'process.exit(0)'], {
      maximum: 2000,
      grace: 1000,
      timeoutDiagnostic: {
        debuggerCommand: root + '/does-not-exist',
        maximum: 100,
        output: root + '/stacks.log',
      },
    });
  assert.equal(result.success, true);
  assert.equal(result.diagnostic, undefined);
  assert.equal(fs.existsSync(root + '/stacks.log'), false);
});

test('timeout captures before graceful termination and still reports failure', async (t) => {
  const root = fixture(t),
    log = root + '/native.log',
    fd = fs.openSync(log, 'wx');
  const debuggerCommand = debuggerScript(
    root,
    "console.log('Thread 1 (native):\\n#0 native_wait ()');",
  );
  try {
    const result = await boundedProcess(
      process.execPath,
      [
        '-e',
        "process.on('SIGTERM',()=>{console.log('stopped');process.exit(0)});setInterval(()=>{},1000)",
      ],
      {
        maximum: 300,
        grace: 1000,
        stdio: ['ignore', fd, fd],
        timeoutDiagnostic: { debuggerCommand, maximum: 400, output: root + '/stacks.log' },
      },
    );
    assert.equal(result.timed_out, true);
    assert.equal(result.success, false);
    assert.equal(result.code, 0);
    assert.equal(result.diagnostic.captured, true);
    assert.equal(result.diagnostic.threads, 1);
    assert.equal(result.diagnostic.frames, 1);
    assert.match(fs.readFileSync(log, 'utf8'), /stopped/);
  } finally {
    fs.closeSync(fd);
  }
});

for (const cancel of [false, true])
  test(`debugger ${cancel ? 'cancellation' : 'timeout'} kills its process group`, async (t) => {
    const root = fixture(t),
      pids = root + '/pids.json';
    const debuggerCommand = debuggerScript(
      root,
      `
    const fs=require('node:fs'),{spawn}=require('node:child_process');
    const child=spawn(process.execPath,['-e',"process.on('SIGTERM',()=>{});setInterval(()=>{},1000)"],{stdio:'ignore'});
    fs.writeFileSync(${JSON.stringify(pids)},JSON.stringify([process.pid,child.pid]));
    process.on('SIGTERM',()=>{});setInterval(()=>{},1000);
  `,
    );
    const controller = new AbortController();
    const start = Date.now();
    const pending = captureNativeStacks(process.pid, {
      debuggerCommand,
      maximum: cancel ? 4000 : 400,
      output: root + '/stacks.log',
      signal: controller.signal,
    });
    if (cancel) {
      for (let i = 0; i < 40 && !fs.existsSync(pids); ++i) await delay(25);
      assert.equal(fs.existsSync(pids), true);
      controller.abort();
    }
    const result = await pending;
    assert.equal(result.captured, false);
    assert.equal(result.cancelled, cancel);
    assert.equal(result.timed_out, !cancel);
    assert.ok(Date.now() - start < 2000, 'debugger completion must not await an unbounded child');
    assert.equal(fs.existsSync(pids), true);
    await assertStopped(JSON.parse(fs.readFileSync(pids)));
  });

test('permission denial and a missing debugger cannot masquerade as captured stacks', async (t) => {
  const root = fixture(t),
    debuggerCommand = debuggerScript(root, "console.error('ptrace: Operation not permitted.');");
  const denied = await captureNativeStacks(process.pid, {
    debuggerCommand,
    maximum: 1000,
    output: root + '/denied.log',
  });
  assert.equal(denied.code, 0);
  assert.equal(denied.ptrace_denied, true);
  assert.equal(denied.captured, false);
  const missing = await captureNativeStacks(process.pid, {
    debuggerCommand: root + '/missing',
    maximum: 1000,
    output: root + '/missing.log',
  });
  assert.match(missing.error, /ENOENT/);
  assert.equal(missing.captured, false);
});

test('capture must fit the existing process termination budget', async (t) => {
  const root = fixture(t);
  await assert.rejects(
    boundedProcess(process.execPath, ['-e', 'process.exit(0)'], {
      maximum: 1000,
      grace: 1000,
      timeoutDiagnostic: { debuggerCommand: 'unused', maximum: 1000, output: root + '/stacks.log' },
    }),
    /reserve/,
  );
});

test('captured frames cannot qualify a probe that crashes instead of exiting cleanly', async (t) => {
  const root = fixture(t);
  const result = await runNativeDebuggerPreflight({
    binary: 'test-owned-probe',
    directory: root + '/probe',
    deadline: Date.now() + 10000,
    execute: async (_command, _args, options) => {
      fs.writeSync(
        options.stdio[1],
        'debugger_attach requested=true permitted=true errno=0\ndebugger-probe ready\n',
      );
      return {
        code: null,
        signal: 'SIGABRT',
        timed_out: true,
        cancelled: false,
        diagnostic: { captured: true, threads: 2, frames: 6 },
      };
    },
  });
  assert.equal(result.target_ready, true);
  assert.equal(result.attach_permission_enabled, true);
  assert.equal(result.complete, false);
  assert.match(result.error, /did not capture permitted native thread stacks/);
});

const nativeSource = String.raw`
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/prctl.h>
#include <unistd.h>
volatile std::sig_atomic_t stopped=0;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t condition=PTHREAD_COND_INITIALIZER;
void signal_stop(int){stopped=1;}
void* worker_wait(void*){pthread_mutex_lock(&mutex);for(;;)pthread_cond_wait(&condition,&mutex);}
int main(int argc,char** argv){
  const bool inherited=argc>1&&std::strcmp(argv[1],"--inherited-probe")==0;
  const bool deny=argc>1&&std::strcmp(argv[1],"--deny")==0;
  if(deny) prctl(PR_SET_DUMPABLE,0);
  else if(!inherited){
    const int result=prctl(PR_SET_PTRACER,PR_SET_PTRACER_ANY);
    std::fprintf(stderr,"debugger_attach requested=true permitted=%s errno=%d\n",result==0?"true":"false",result==0?0:errno);
    if(result!=0)return 2;
  }
  if(argc>1&&std::strcmp(argv[1],"--exec")==0){
    char* args[]={argv[0],const_cast<char*>("--inherited-probe"),nullptr};
    execv(argv[0],args);return 3;
  }
  if(inherited) std::fprintf(stderr,"debugger_attach inherited=true\n");
  std::signal(SIGTERM,signal_stop);std::signal(SIGINT,signal_stop);
  pthread_t thread; if(pthread_create(&thread,nullptr,worker_wait,nullptr)!=0)return 4;
  std::fprintf(stderr,"debugger-probe ready\n");std::fflush(stderr);
  while(!stopped)usleep(10000);
  return 0;
}
`;

test('host gdb captures native pthread stacks, preserves attach permission across exec, and reports denial', async (t) => {
  const debuggerCommand = process.env.BLITZ_TEST_NATIVE_GDB ?? 'gdb',
    compiler = process.env.CXX ?? 'c++';
  try {
    execFileSync(debuggerCommand, ['--version'], { stdio: 'ignore', timeout: 5000 });
    execFileSync(compiler, ['--version'], { stdio: 'ignore', timeout: 5000 });
  } catch {
    t.skip('Host GDB and C++ compiler required; exercise explicitly with the pinned debugger');
    return;
  }
  const root = fixture(t),
    source = root + '/hang.cpp',
    binary = root + '/hang';
  fs.writeFileSync(source, nativeSource);
  execFileSync(compiler, ['-std=c++20', '-O0', '-g', '-pthread', source, '-o', binary], {
    timeout: 15000,
  });
  const preflight = await runNativeDebuggerPreflight({
    binary,
    directory: root + '/preflight',
    deadline: Date.now() + 10000,
    debuggerCommand,
    diagnosticMaximum: 4000,
  });
  assert.equal(preflight.complete, true, JSON.stringify(preflight));
  assert.ok(preflight.process.diagnostic.threads >= 2);
  assert.match(fs.readFileSync(root + '/preflight/threads.log', 'utf8'), /worker_wait/);
  for (const [arg, expected] of [
    ['--exec', true],
    ['--deny', false],
  ]) {
    const output = root + '/' + (expected ? 'exec' : 'denied') + '.log';
    const result = await boundedProcess(binary, [arg], {
      maximum: 500,
      grace: 5000,
      timeoutDiagnostic: { debuggerCommand, maximum: 4000, output },
    });
    assert.equal(result.diagnostic.captured, expected, JSON.stringify(result));
    if (expected) assert.match(fs.readFileSync(output, 'utf8'), /worker_wait/);
    else assert.equal(result.diagnostic.ptrace_denied, true);
    await assertStopped([result.diagnostic.pid]);
  }
});
