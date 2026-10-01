# Local validation before the hour retry

The local build completed after capability fixture fix `661a01c`; CMake
regenerated and Ninja reported no work to do. The bulk CTest run passed 49/49
tests in 64.26 seconds, explicitly excluding `neural-native-stack-contracts`.
Intervening commit `cc706ea` only preserved the failed remote attempt's evidence.

After native-stack fix `959f248`, that excluded CTest passed separately (1/1 in
3.73 seconds). The union of the two JUnit test-name sets matches all 50 configured
local CTests. This is coverage across two source-scoped runs, not one simultaneous
50-test execution. No C++ runtime source changed between them; the intervening
executable/test changes affect only the separately rerun native-stack contract.

The capability orchestration tests also passed 25/25 from each of three working
directory classes: source, build and unrelated. The controller supplied those
classes; the logs do not contain the absolute working-directory paths. Their raw
Node timings are preserved without normalization.

[record.json](record.json) scopes revisions, test names and limitations.
[manifest.json](manifest.json) checksums the original build, CTest/JUnit and Node
logs. The [exact Ubuntu debugger reproduction](../native-stack-denial/README.md)
is separate evidence. These local passes do not replace the failed remote result,
or establish training-hour completion, model quality or throughput. The next
remote bundle is frozen at `959f248`.
