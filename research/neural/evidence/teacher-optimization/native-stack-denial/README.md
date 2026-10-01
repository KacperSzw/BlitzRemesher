# Ubuntu GDB denial-classification regression

The remote engineering gate failed its expected-denial assertion after positive
native stack capture had succeeded. This local CPU-only reproduction used the
exact Ubuntu GDB 15.1 package installed remotely. As UID0 without CAP_SYS_PTRACE,
it printed explicit Yama permission advice followed by the misleading errno
`ptrace: Inappropriate ioctl for device.` The previous parser missed that advice.

[Before](before.log) passes 7/8 tests and reproduces the same denial-classification
failure. The copied pre-fix test differs only by including raw debugger output in
its assertion message. [After](after.log) passes 9/9, with no skips. The fix
recognizes the precise Ubuntu warning; unrelated ioctl and missing-process errors
remain unclassified failures. Positive capture, two-thread/worker-frame evidence,
attach permission across exec and clean preflight exit requirements are unchanged.

[record.json](record.json) records source/package hashes, the official download
URLs, credentials and command. Both Debian packages were extracted under
`/tmp/blitz-gdb15`; existing Nix shared libraries let that unmodified debugger
run on NixOS. [The wrapper](gdb-wrapper.cjs) and its
[library environment](gdb-wrapper-environment.json) record that local setup.
Root credentials applied only to disposable test processes; no system setting
was changed. No GPU, provider or training operation was performed.

This is a parser regression check, not a new remote validation result or neural
quality measurement. The historical remote failure remains intact.
