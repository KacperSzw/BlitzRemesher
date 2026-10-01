# First remote teacher optimization attempt

The A40 run `teacher-optimization-02` used frozen source `5244fb3`. It is an
**incomplete experiment**, with no accepted performance score, strategy adoption,
paired learning pilot or full quality result. See [record.json](record.json) for
counts, revisions, limits and cleanup, and [manifest.json](manifest.json) for
SHA-256 checksums of every stored artifact.

Remote CTests passed 43/43. Action-GPU and grouped Vulkan Compute Sanitizer checks
reported zero errors; Vulkan validation passed. The separately uploaded CPU
cancellation contract passed without modifying the frozen source bundle.

The baseline process completed. All 11 jobs in the first optimized process also
completed and were consumed, but that process stalled afterward and was killed
by the deadline. Its last report cannot distinguish worker shutdown, main-stream
synchronization or enclosing scope cleanup. Only two of the four planned ABBA
processes ran, so the completed job timings do not establish a remote speedup.

[parity-proof.json](parity-proof.json) records independent checks of all 11 job
pairs against the actual extracted action/episode files and JSON: payload bytes,
normalized exhaustive contracts, every common trajectory field, semantic index
fields and reuse counters match. Only the documented version-5-to-6 exhaustive
contract migration and additive `candidate_search` trajectory field are
normalized. `job-artifacts.json.gz` preserves the original text and SHA-256 of
all 88 index, contract, trajectory and reuse files. The two process reports and
logs, experiment report and contract reports are stored verbatim as gzip files.

The 137,184,725-byte results archive remains in the ignored run directory; its
verified SHA-256 is
`65b8cfda53e4d4fe6550da911f29635de9a19b0b6163a39507c0d39a54b5e147`.
Collection preceded termination. The later [provider readback](provider-cleanup.json)
contains no Pods or network volumes. `rental.json` omits only the connection
endpoint; the manifest retains the original ledger checksum. No credentials or
SSH private identity are included.

The conservative grant cost is $0.1775 rounded up, using elapsed rental time and
the GPU cap plus storage allowance. It is not an invoice. Remaining authorization
permits bounded diagnostics; shutdown regression and complete uncensored quality
remain prerequisites for promotion or final training.
