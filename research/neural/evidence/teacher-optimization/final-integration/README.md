# Final local integration and readiness evidence

This archive records validation of source through `17294cb` after the requested
remote hour. No learning, provider operation or new rental was performed by
these local checks. [The current review](../../../TEACHER-OPTIMIZATION.md)
separates completed engineering validation from the failed quality gate.

`manifest.json` hashes each compressed payload and its original bytes. Large
executables, model weights, training tensors and source mesh payloads remain in
ignored local storage; their identities are preserved in the reports. Paths in
raw reports identify the original workstation inputs.

## Integrated contracts and corrected quality

`validation/` contains the affected-target build, complete native-target build,
CTest log and JUnit XML. All 53 configured CTests passed with no skips in 61.24
seconds. Relevant focused CPU sanitizers and GPU memchecks are preserved in the
separate [audit validity](../audit-validity/README.md),
[optimizer validation](../optimizer-validation/README.md), and
[GPU admission](../action-admission/README.md) archives.

`corrected-quality/` preserves both native audit outputs, the comparison report,
settings, asset manifest, context and exact verification script. The initializer
and final-hour checkpoint were rerun after the runtime corrections. All 48
source/adjacent measurements passed, with no resource, cancellation or
confirmation failures. Every non-timing output field is exactly equal to the
corresponding [frozen-runtime result](../remote-hour-413184/README.md). Only the
binary digest and the individually named timing fields are excluded from that
comparison. This includes equality of counts, visual measurements, proposal stop
reasons, diagnostics and storage; it does not assert saved geometry byte equality.

There are 35 trial-budget stops, eight target stops and five no-accepted-action
stops across both checkpoints and ranking controls. This small coverage-only
diagnostic has no aggregate SCORE or release claim. The shelves have mixed
per-level changes, and final rock LOD3 remains 1,134 triangles versus the
initializer's 930. The correctness fixes do not remove this observed regression.

## Actual hour supervision and timing

`hour-analysis/` contains the per-shard checksum ledger, aggregate label counts,
timing summary and independent verification receipt. The verification script
checks all 3,228 consumed shards against the full local collection: every shard
is complete, trajectory length equals its state count, and preferred edge IDs
are a subset of queried edge IDs. Their state count equals `latest.json`'s 53,260.

31,824 states have four preferred edges out of four queried edges. Thus at least
59.7522% of states have no preferred/nonpreferred ranking pair. This does not
mean zero placement/pass/shared-backbone learning and does not justify arbitrary
tie breaking. It also does not measure classification accuracy.

The timing analysis re-sums the actual `history.jsonl`: 3,229 teacher phases
(including one interrupted at the hour deadline), 3,228 training segments,
3,496.0407 seconds of learner wait and 83.5511 seconds of optimizer updates.
The denominator is the 3,600-second learning interval. Checkpoint counters agree
with `latest.json`. This establishes a bottleneck in this run, not a cross-run
performance comparison. Raw hour history and final metadata are also preserved
in the separate remote-hour archive; the complete collected tarball remains
local with its verified SHA-256.

## Correctness baseline and unlaunched preparation

`baseline-v3/` contains overlay provenance, successful native build and 66/66
focused Node tests. The old v2 Ubuntu/A40 cache is explicitly rejected before
environment inspection and never restored. V2 definition and patch hashes
remain unchanged. The v3 baseline keeps the original serial teacher and applies
only reviewed lifecycle/seed/admission corrections, with no optimized raster
reuse transplanted into the baseline.

The two ABBA comparisons each have four successful processes and 16 jobs,
including eight warmups. Independent verification invokes the native-output
contract validator, rehashes actual action/episode bytes and compares every
normalized output and common trajectory across all four processes. All match.
Each run has two source-only conditions and one worker under a 1,024 MiB budget.

Baseline-v3/current warm wall ratio is 1.0680. Pre-correction/current ratio is
0.9690, or approximately 3.20% more warm time for the corrected teacher. The
shared RTX 2080 and concurrent CPU compilation make both diagnostic measurements.
No isolated speedup, large-corpus extrapolation, paired learning or quality
improvement is inferred. Raw timings, traces, counters and logs remain visible.

`preparation/prepared.json.gz` describes the fresh no-cache archive. Independent
verification checked all 30 input files, the 167,402,987-byte gzip archive and
its source bundle, frozen at `17294cbe8dfeaf112b2ca7f57088e210ac5ad94f`. Archive
SHA-256 is `9493661cada81ce59fd4dcd6f766ade3e3813353816bb5379a462b2c4a3a880f`.
The source bundle intentionally names the frozen commit as `HEAD`; an initial
verification-script assumption that it exported `refs/heads/main` failed and was
corrected after inspecting the bundle. The exact commit and prepared branch
are both checked. No rental file or baseline-cache payload exists.

This preparation is reproducible infrastructure evidence. The remaining local
quality work changes the experiment; regenerate its bundle before a later
authorized launch rather than blindly launch this frozen hour recipe.

## External review disposition

Three attached-source Prophet reviews returned: objective/quality, optimizer
loss/checkpoint handling, and LOD admission. The exact objective response is
preserved in [hour-quality diagnostics](../hour-quality-diagnostics/README.md).
The two later responses were received in tool output but their exact response
files were not retained. The following is an attributed summary, not a quote:

- Loss/checkpoint review identified nonfinite loss settings and negative Adam
  second moments. Both were reproduced and fixed in `f822850`. Valid-label
  gradient/layout checks did not reveal another concrete defect. A speculative
  malformed-label concern was checked against the existing compact/dense input
  validation; no redundant validation layer was added.
- Admission review identified stale geometry classes after coincident placement,
  missing initial precision-cap admission, inaccurate CPU stop reasons and
  inconsistent numerical audit predicates. Local regressions substantiate the
  fixes in `39b2de7`, `e4613db` and `276c83c`. Speculation about UV relaxation,
  normal condition channels and omitted fallback source was not promoted to a
  confirmed defect.

`prophet/` preserves the exact three additional source attachments and file-hash
manifests. The optional fourth review, covering ownership, had not returned after
approximately an hour. Its waiting execution was terminated without a result;
it is not counted as a passed review. Local tests and reproduced failures, not
external review assertions, are the correctness evidence.

`verification/` contains the CPU-only evidence-verification scripts and the
bounded local ABBA runner. Run verification from the repository root after
restoring the original local artifacts; these scripts never start a rental.
