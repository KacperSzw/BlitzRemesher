# Second remote teacher optimization attempt

The A40 run `teacher-optimization-03` used frozen source `ec9660f`. Setup and both
native builds completed, but the teardown fixture timed out before the contract
suite, teacher comparison, paired learning or quality stages. This is a failed
diagnostic attempt, with no accepted speedup, quality score or strategy adoption.
See [record.json](record.json) for the observed boundary and
[manifest.json](manifest.json) for checksums of the preserved text artifacts.

The native debugger preflight passed. The teardown fixture completed rounds 0
and 1, then stalled in round 2. Ordinary GDB successfully captured 10 threads and
86 frames before the bounded runner terminated the process with SIGTERM. The
original trace and capture are preserved in `teardown-stress.log.gz` and
`teardown-stress.threads.log.gz`.

One application worker finished its logged Vulkan, session and stream resource
destruction, then appears in NVIDIA EGL/GLSI thread-exit cleanup. The other
application worker remains inside `vkDestroyDevice`. A driver-internal thread
also appears in thread-exit cleanup. This establishes the observed overlap; it
does not prove the proprietary lock cycle, an application ownership bug, or that
serializing C++ destructors alone would fix it.

The injected [vendor ICD](nvidia-icd.json) names `libGLX_nvidia.so.0`; frozen cloud
setup rewrote it to the [selected EGL ICD](nvidia-headless.json). Both the
[GLX dependency report](nvidia-glx-dependencies.txt) and
[EGL dependency report](nvidia-egl-dependencies.txt) resolve all listed libraries.
NVIDIA's documentation for
[driver 580.159.04](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.04/README/installedcomponents.html)
supports both as Vulkan ICDs: GLX is the vendor default, and EGL is recommended
when X11 client libraries are unavailable. Forced EGL selection is therefore a
candidate environment cause at the time of this attempt. The later
[local GLX reproduction and retirement controls](../local/lifecycle-validation.json)
show that ICD selection alone is insufficient for the exercised fixture. They
preserve this failure and do not establish an exclusive proprietary root cause.

The 37,156-byte results archive remains in the ignored run directory. Its verified
SHA-256 is `e488deb0204fcedcd64130bf4db761bc00ba3f1d14a3772da0f6e5fbe22e979c`.
Collection was verified before compute termination at 12:08:16 UTC on 2026-10-01.
The independent [provider readback](provider-cleanup.json) at 12:11:56 UTC showed
zero Pods and zero network volumes. The copied rental ledger omits its connection
endpoint; no credentials, SSH identity, binaries or archives are committed.

The conservative cost is $0.1405 rounded up for this attempt and $0.3180 rounded
up cumulatively across both attempts, using elapsed time at the GPU cap plus
storage allowance. These are grant estimates, not provider invoices. Remaining
authorization permits bounded diagnostic controls. Normal representative teacher
shutdown and complete uncensored development quality remain prerequisites for
promotion or final training; exhaustive teaching remains selected.
