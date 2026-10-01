# Native teacher timeout diagnostics

The diagnostic retry sets `timeout_diagnostics: true` in the frozen teacher
optimization request. Ordinary `boundedProcess` calls do not start a debugger.
The option does not permit a longer rental or extend any experiment deadline.

Before GPU work, the runner starts the current placement preparation executable
with `--debugger-probe`. This CPU-only process explicitly permits debugger
attachment, announces readiness, and waits for termination. The runner captures
host thread stacks and requires the probe to exit with code zero after SIGTERM.
Denied attachment, missing frames, debugger failure, a crashed probe, or an
expired deadline stops the experiment before GPU stress, profiling or learning.

The retry uses ordinary `gdb`, installed by the cloud setup. Local CUDA-gdb 12.9
crashed while injecting initialization into the CUDA-linked probe, and crashed
the probe itself. CUDA-gdb is therefore not the default for this diagnosis.
The debugger command remains configurable for separately validated tools.

Two independent native opt-ins are used:

- `BLITZ_ALLOW_DEBUGGER_ATTACH=1` permits same-user attachment using Linux
  `PR_SET_PTRACER_ANY`. It does not bypass container seccomp or stronger Yama
  restrictions. A failed permission call or attach is reported explicitly.
- `BLITZ_TEARDOWN_TRACE=1` emits final teardown boundaries to stderr. It does
  not grant attachment permission or trace the measured warm work.

The immutable baseline is started through the current executable's
`--debugger-exec EXECUTABLE ARGS...` mode. Permission survives `execv`, while
`/proc/self/exe`, the baseline binary digest and teacher contract still identify
the original baseline. The profile records the launcher digest and diagnostic
flags. Process wall ratios are reported as `diagnostic_process_wall_ratio`,
since startup and teardown now include instrumentation.

After successful attachment preflight, the remote runner executes the candidate
Vulkan test's `--teardown` mode with a 45-second workload budget. Its capture and
termination allowance remains inside the existing teacher phase deadline.
The same timeout capture is enabled for resident benchmarks and paired cycles.

On a timeout, the target remains alive while a separate debugger process group
runs `thread apply all bt 32`. Arguments, locals, auto-loaded scripts and
debuginfod lookup are disabled. The debugger has a five-second deadline; a
six-second termination allowance bounds capture plus target termination. The
target's SIGKILL timer is armed before capture, and a hung debugger and its
descendants are killed independently. Cancellation aborts capture and signals
the target immediately. Capturing stacks never makes a timed-out run successful.

The debugger result records attempted/captured status, exit status, deadline,
permission denial, frame/thread counts and the stack log digest. Missing or
partial evidence remains a diagnostic failure, not a successful stack capture.

CPU-only contract tests include a compiled pthread hang, successful host-stack
capture, attachment permission retained through exec, explicit permission
denial, target crash rejection, and cleanup of a hung debugger's descendants:

```sh
BLITZ_TEST_NATIVE_GDB=/absolute/path/to/gdb \
  nix develop .#neural --command node --test tests/neural_native_stack.mjs
```

The native capture test reports a skip if its C++ compiler or host debugger is
unavailable; the diagnostic remote preflight always requires successful capture.
