# GPU action representation admission

Local deterministic regressions confirmed two representation bugs and validate
the fixes in `39b2de7f11579ae18a0c6105e97f5b5312af9e9f`. This is correctness
evidence, not a new training, quality or teacher-throughput result.

A legal edge in one of two disconnected components could place its survivor at
the other component's vertex. The committed state retained 12 legal actions;
reconstructing it from its snapshot yielded only 4. The fix rejects new
coincidences between distinct live geometry classes. Original class identities
and attribute charts stay stable. Same-class seam wedges and discarded source
positions remain allowed. The tests cover FP32 and packed positions, signed
zero, per-candidate batch validity, rejected-commit immutability and complete
directed endpoint-position inventory equality after an accepted snapshot.

The initial state also admitted 6 exact vertices out of 100 under a 5% cap.
Constructor admission now checks unique referenced IDs after optional UV
welding, including episodes that never attempt an edit. Tests use explicit
source quantization bounds: 5/100 passes, 6/100 fails at 500 basis points and
passes at 600. Unreferenced flags neither count as exceptions nor increase the
denominator. UV welding can reduce that denominator and must be accounted for.
Validation uses scratch bits and preserves supplied/reset data.

The affected target built successfully. The final full
`neural-action-gpu-contracts` CTest passed 1/1 in 1.48 seconds, with no skips.
Compute Sanitizer memcheck reported zero errors. An independent source review
found no concrete blocker. Tests ran on a shared RTX 2080 with driver 595.71.05;
unrelated applications were left running.

## Cost and retained negative result

The new guard allocates no memory. Each candidate clears the existing hash
allocation and hashes three corners per surviving face inside the existing
placement-validation kernel. The clear writes
`4 * next_power_of_two(2 * allocated_vertices)` bytes: 64 KiB in this fixture.
Initial precision admission adds a referenced-index pass and bitmap reduction
only when a precision bitmap exists.

A first implementation reused the constructor's hash unchanged. Its low bits
cluster integer-grid coordinates: all 4,225 vertices in the timing fixture
started in one bucket. Median time for 64 serial trials increased from 3.12 ms
to 667.89 ms. That implementation was rejected; its raw timings remain here.
The final guard mixes high bits before selecting a bucket. The original
constructor hash is unchanged and remains a separate optimization opportunity.

The final paired local sequence repeated one valid endpoint trial on an
8,192-face plane, with no commits, audits or learning. Four runs per version
gave serial medians of 3.31 ms before and 4.60 ms after per 64 trials, about 39%
more in this small case. For batches of four, final times were 12.44–13.66 ms;
baseline times ranged from 7.38–26.40 ms, too noisy for a useful ratio. These
short measurements include host synchronization and shared-workstation noise.
They do not measure the full teacher, large corpus inputs, or LOD quality.

## Reproduction and provenance

Build the affected target in `nix develop .#neural`, then run:

```sh
ctest --test-dir build/neural -R '^neural-action-gpu-contracts$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/neural/blitz-neural-action-gpu-tests
build/neural/blitz-neural-action-gpu-tests --placement-coincidence
build/neural/blitz-neural-action-gpu-tests --initial-precision
build/neural/blitz-neural-action-gpu-tests --placement-timing
```

The two selectors failed before the fix and passed afterward. Their standalone
post-fix logs precede the hash-mixing refinement; final full-suite and memcheck
logs cover the committed implementation and explicit quantization fixtures.
The final timing binary precedes only that test-fixture strengthening; its
runtime and timing workload match the committed source. Binary digests in the
raw timing records distinguish each version.

`record.json` records exact source/binary hashes, test results and measurement
limits. `manifest.json` checksums both compressed and uncompressed raw files.
The preserved dirty-state record includes other agents' concurrent changes;
only the two named source files belong to this fix. The timing path calls
`GpuActionState` directly and does not invoke the concurrent CPU/audit changes.
Native binaries remain in ignored local storage rather than this archive.
