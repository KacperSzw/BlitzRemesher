# Local policy-alignment investigation

## Current decision: finish the support diagnostic and hand off

The user explicitly requested the simpler score-one/apply-one/rescore loop on
2026-10-01, followed by remote training only after positive local validation.
The new remote authorization is at most $10 including setup, collection and
retries. It does not authorize renting before the local gate. On 2026-10-02 the
user asked to finish the current part, push main and let another agent continue.
That supersedes this agent's earlier instruction to continue through rental
launch; the local success condition for future paid training remains unchanged.
Coverage and both UV modes come first; shading remains a later gate.

The sequential experiment was `policy-ranking-13`, configured by
`policy-ranking-sequential.json` and `sequential-teacher-quality.json`:
single-collapse endpoint execution, 1,024 trials per proposal, four training
parents at two source sizes, both UV modes, fresh native trajectory labels,
three 512-update training seeds, and unchanged independent diagnostic assets.
It isolates the 65 ranking parameters; it does not prove that training
the whole model would or would not work. If the frozen representation limits
the test, report that explicitly rather than extrapolating a capacity verdict.
Positive teacher health or training loss alone cannot open the remote gate.

The interrupted run left fifteen complete shards and one native report failure:
`selection requires 2..32 levels`. A cancelled source-only result was valid, but
its JSON exporter attempted a chain-selection sweep requiring two levels.
The deterministic regression failed before the exporter guard and passed after
it, including ASan. The missing BlueCrab/UV-off/2,048 shard was regenerated with
the original pinned executable and unchanged visual/action settings; only the
duration allowance increased from ten to twenty minutes. It completed in
232.153 seconds. All sixteen verified shards are copied into
`sequential-dataset-01`, with 2,173 observed and 1,720 informative states.

`policy-ranking-14` failed its CUDA memory query before any update. After memory
became available, all three seeds in `policy-ranking-15` completed 512 updates
with finite loss, exact optimizer restoration and exact native model export.
The first completed two-asset diagnostic (UV on, seed 101) nevertheless worsened
category-balanced retained ratio by 3.0206%; its worst individual LOD increased
by 5.4945%. All visual gates passed. This is a narrow negative diagnostic, not a
complete experiment score. Further audits of that restricted readout were
stopped explicitly to prioritize the capacity test.

The capacity follow-up is implemented in `endpoint-ranking-sequential.json`.
It keeps the v4 feature layout and trains both hidden layers plus the ranking
output: 12,481 parameters instead of 65. It has a first-class
endpoint-only model scope, enforced by the native generator, because changing
the shared representation also changes the existing placement outputs even
when their final weights are frozen. The exported scorer must reject Rebuild;
the old placement model remains separately valid. BLZNET03 stores the endpoint
scope; legacy BLZNET01/02 remain unrestricted. Endpoint optimizer checkpoints
use version 3 while old joint/ranking-row version-2 interpretation is preserved.
Admission binds verified version-10 endpoint teaching to its original collector
payload, requests, trajectory, rejection bitmap and exact source reference.
CPU model/data tests, ASan and both eager/captured Adam restoration tests passed.
This is a capacity test, not a demonstrated quality improvement. No endpoint
model has been admitted for remote training yet.

`endpoint-ranking-01` completed the three fresh full-scorer fits: losses fell to
0.0127, 0.0415 and 0.0309, with 97.5--97.8% preferred membership and exact native
exports. Its first complete UV-on seed-101 diagnostic was negative: rock
1,638/820/410 became 1,936/968/484, while shelves 117/97/84 became 116/97/84.
Category-balanced retained ratio worsened by 10.8215%, with an 18.1929% worst
LOD increase. The remaining audits were stopped explicitly. The partial result
has no experiment/pilot score and does not open the remote gate.

The implementation review found the fresh training path sound, but identified
four follow-up contracts: native weight refresh must carry use scope; model-only
checkpoint loading must preserve endpoint scope; data provenance must bind the
shard index (including sampler categories); and low-level endpoint updates must
require the conditioned architecture, not just its shared input width. These
are fixed and covered by CPU, native CUDA and sanitizer checks. An asynchronous
endpoint-to-ranking-row checkpoint also preserves the endpoint restriction.

One preselected unseen training rock (`ph_coast_line_02`, UV-on, 512-source)
produced complete verified teaching in 25.487 seconds. A descriptive CPU double
probe found perfect preferred membership for all trained models there, but no
fully-passing, same-removal margin pairs. This fixture therefore does not test
preservation discrimination. The first diagnostic parser omitted the bitmap's
uint64 vector length; its invalid result is retained and explicitly superseded
by the parser with header, size and native eligible-state-count assertions.

`endpoint-plane-probe-01` completed both UV modes under the same single-action
budgets. The existing classical current-plane comparator produced rock
1,256/628/314 in both modes, versus neural initializer 1,638/820/410 (UV-on) and
2,692/1,346/674 (UV-off). Some shelves levels regressed, so neither the comparator
nor these two assets establish a complete quality success.

The predeclared diagnostic follows Prophet's concrete support hypothesis:
the teacher labels at most 16 mixed-policy/geometric actions, but deployment
sorts thousands of legal actions. Keep the trained seed-101 scorer, collector
initializer, UV-on rock, target, cameras and 1,024-trial ceiling fixed. Restrict
learned choices to the same native 16-action PolicyMixed selector constructed
by the frozen collector; no additional teacher audits and no target/loss/feature
changes. Log global and supported top actions/scores and support escape at each
rescore. Recovery of all three rock levels within the already-declared +2% cap
would make support mismatch a sufficient explanation for this regression.
Failure would keep objective, representation and state-distribution hypotheses
open. This diagnostic alone cannot pass the full frozen pilot or authorize
paid training. Do not collect the remaining 32 diversity shards first.

`endpoint-support-01` completed the UV-on seed-101 rock diagnostic in 56.714
seconds, with all exact final visual gates passing. The supported trained chain
was 1,256/628/314 triangles, recovering all three levels under the predeclared
+2% cap and beating the unrestricted initializer by about 23%. The unrestricted
trained top action lay outside the collector pool in 3,705 of 3,709 rescoring
events (99.892%). This supports the candidate-support explanation for the
observed regression, not a general quality claim. The matching supported
initializer (`endpoint-support-initial-01`) completed in 70.143 seconds with
the same 1,638/820/410 counts as its unrestricted baseline and zero support
escapes. Collector, cameras, source and executable hashes match
between the supported comparisons. Under that identical restriction the
trained scorer improves every rock LOD by about 23%, giving a local conditional
learning signal. One seed, one rock and UV-on do not pass the frozen pilot.
Both diagnostics use collector inference and
geometric ordering in addition to candidate inference; their work differs from
ordinary neural deployment and is recorded explicitly.

The first support implementation read a stale host action count before ranking
refreshed GPU state after a commit. The existing observer cancellation-poll parity
test failed. Reading the unrestricted count after ranking fixes it without
changing the assertion. The new fixture also rejects all 16 supported actions
and verifies that execution stops instead of falling through to other legal
global actions, with no committed geometry change.

The sequential runtime now uses the conservative sparse certificate for each
action's four source/previous/search/audit gates. Final emitted measurements and
dense teacher margins remain exact. `sequential-gate-parity-01` completed on the
two diagnostic assets in both UV modes: ordered trial traces, final FP32 mesh
hashes and every exact final error matched the original path, with no unknown,
cancelled or resource-limited evaluation. Audit time decreased. Shared GPU work
overlapped the UV-on old-path measurement, so those timings cannot establish a
controlled speedup factor.

The read-only activation inspection found no nonfinite or oversized inputs and
no all-zero hidden embeddings in the first seven completed sequential shards.
The UV input weights are indeed zero and frozen in the restricted experiment.
Between 23 and 29 of the 64 second-layer units never activate on an individual
inspected shard. These observations do not prove that representation capacity
caused a product regression. The exact files, hashes and CPU diagnostic are
retained in `endpoint-representation-01`.

Before this experiment, `single-action-diagnostic-01` compared saved weights
using batch one and 1,024 trials, giving the same maximum selected-action
capacity as 64 trials of batches up to 16. Audit work/time is not equal.
The initializer's UV-on rock chain improved from 1,862/932/930 to
1,652/826/766 reduced-LOD triangles. The saved early seed 101 instead produced
1,898/950/908; therefore the runtime change alone did not demonstrate useful
learning. The old-checkpoint sweep was deliberately stopped during seed 211
to prioritize collecting fresh sequential data. Its incomplete raw results
and stop decision remain available, without an aggregate score.

## Findings from focused reviews

Four independent Prophet reviews examined the objective, optimizer, runtime
search and observability. A follow-up corrected the optimizer review: Adam's
coordinatewise preconditioning can move parameters outside the span of raw
pair gradients even without a logit penalty. Such a projection is not proof
that regularization caused the regressions. Absolute-logit regularization is
still a testable concern, not an established cause.

Read-only analysis of experiment 09 found that 416 of 715 informative states
only distinguish removal of two faces from removal of one, with equal measured
quality. This represents 38.42% of actual state-sampler probability; cardinality
pairs consume 43.97% of pair-loss probability. Only 110 states have preferred
equal-removal margin pairs. Exact feature comparison found no reversed
preferences after ignoring the inactive UV input. Same-removal, fully passing
quality-pair accuracy after long training is about 55% UV-on and 67% UV-off,
averaged equally over states. This is much weaker evidence of preservation
learning than 95% preferred membership. Raw scripts and hashes are retained.

The runtime scheduler previously spent a two-proposal budget on two origins of
one output mode. The corrected diagonal schedule covers both modes and origins
before additional parents. A deterministic CPU fixture failed before the fix
and now emits 8/4/2/1 triangles. Re-evaluating all three early heads and one
long-trained head with matching corrected initializer/constant controls still
showed quality instability. Those complete results are in
`execution-diagnostic-01`.

Optional native execution traces locate a concrete failure mechanism: the bad
early UV-on rock seed rejects 33 of 64 trials at the source audit, repeatedly
halving 16/8/4 to accept two collapses. The two better seeds reject only 1–3
trials on that request. No rejected singleton is needed to explain the bad
seed. Traced/untraced contract tests preserve meshes and work counters, and
CUDA memcheck reported zero errors.

Endpoint teacher version 10 (`runtime-endpoint-v3`) also includes both search
changed-area terms and excludes preferred actions that cross the runtime
triangle target. Pixel measurements are retained honestly even for a
target-ineligible action. Adding the two area terms changed zero preferences
in the 856 old states, so that omission is not the cause of those regressions.
All three saved run-09 step-512 payloads exactly match their run-10 counterparts.

A counterfactual continuation prototype was retained outside the runtime and
deferred in favor of the user's sequential baseline. No new continuation or
reinforcement-learning machinery is required before testing this simpler path.

The first remote-readiness gate covers coverage and both UV-preservation modes.
Normal/shading quality follows after this gate, as requested. Paid training is
conditional on the local success gate above. Local timings include other
workstation users.

The hypothesis is that an edge label must describe the action the runtime will
actually execute. Oracle placement quality cannot establish the quality of a
different learned placement on the same edge. The controlled first experiment
therefore freezes the pretrained placement policy and trains its last ranking
row only. This deliberately tests label alignment before adding model capacity.

## Contracts

- Teachers audit the native decoded placement, including source-domain position
  quantization and UV conditions, with the configured source and adjacent views.
- Native geometry rejection is a separate one-bit observation. It is a known
  infeasible ranking alternative, never a fabricated visual measurement.
  Incomplete visual observations do not become negatives.
- FP32 feature storage avoids changing the policy action between label generation
  and training. Dataset admission binds the actor payload, semantic teacher
  version, raw-data checksum and geometry bitmap checksum.
- Ranking-row experiments train only that row and bias; endpoint-scorer
  experiments also train the trunk. Frozen output rows and their optimizer
  moments remain unchanged. Endpoint exports therefore carry a Reuse-only
  scope. Save/restore, captured/eager execution and native export are checked
  independently.
- All tied/non-preferred states remain in raw data. The ranking sampler excludes
  states with no usable preference pair and reports their count.
- Initial and trained models use identical evaluation inputs, settings, cameras
  and work limits. Constant-ranking controls must be identical except for timing.
  Every source/adjacent measurement must be finite and inside its configured
  pixel and area limits. Scheduled duplicates and regressions remain visible.

## Experiments retained locally

`scripts/neural/policy-ranking.mjs CONFIG FRESH_OUTPUT` runs a bounded local
experiment and saves its inputs, source and binary hashes, teacher records,
checkpoint health, full audits and comparisons. It has no provider actions.
`dataset_directory` explicitly reuses verified teacher data for an optimization
experiment. These are diagnostics, not the release protocol SCORE.

| Directory under runs/neural | Question | Observed result |
| --- | --- | --- |
| policy-ranking-01 | Can native placement labels and rank-only updates be produced? | First 32-state shard: 29 informative states, 1,158 pairs, 53 geometry rejections. Engineering smoke only. |
| policy-ranking-02 | Two training meshes, two screen sizes; 2,048 updates, three seeds | Complete but negative independent-mesh quality. Rock regressions; no readiness claim. |
| policy-ranking-03 | Four categories and matched LOD scales/limits/audit cameras; 8,192 updates | 1,152 states, 754 informative. About 96% training preferred membership, but independent-mesh results vary strongly across seeds and UV modes. |
| policy-ranking-04 | Same data, 32,768 updates and batch 256 | Training preferred membership about 97%; quality still inconsistent. More repeated updates did not solve the problem. |
| policy-ranking-scope-01 | Isolate rebuilt placement output using experiment 03 models | UV-on placement quality did not improve consistently. The automatic-mode improvements predominantly involved endpoint reuse. |
| policy-ranking-05 | Change only teacher candidate selection to policy-mixed | 1,152 states, 871 informative. UV-on rock still regressed for every seed; UV-off category-balanced retention worsened for every seed. Sampling alone was insufficient. |
| policy-ranking-06 | Observe actual runtime endpoint reuse, conditions and predecessor | 384 states, only 100 informative (96 bolt cutters, 3 onion, 1 boulder, 0 stress). All eight shards preserved native output and action counts exactly. Training membership reached about 98%, but independent shelves barely reduced and every seed/mode comparison failed. |
| policy-ranking-07 | Add 512/2,048-triangle source variants | Stopped after eight completed shards when missing search gates were established. Partial data retained; no learning or quality conclusion. |
| all-gates-control-01 | Re-evaluate unchanged initializer and experiment-06 checkpoints with all runtime gates | Complete, three seeds/both UV modes. Initializer UV-on shelves improved from 518/259/247 to 262/131/115 triangles. Old trained checkpoints still regressed in every comparison; this is a runtime repair, not successful new learning. |
| policy-ranking-08 | Corrected gates, evenly sampled trajectories, rebuilt 512/2,048-triangle sources | Stopped before learning: onion support was insufficient and the 512-triangle mountainside variant had no legal actions. Eight completed shards and the empty rock observation are retained. |
| reuse-source-{rock,onion,crab}-01 | Preserve original endpoint attributes when preparing source variants | Legal teaching actions returned, but reduction stopped at 6,352/1,290/35,574 triangles. The crab yielded only one informative state out of 192. These sources do not solve sparse supervision. |
| merged-source-rock-01 | Use existing chart-aware wedge merging during source preparation | Reached 512 triangles; 39 of 51 observed states informative, 1,133 pairs, exact native execution parity. This verifies useful data, not learned quality. |
| policy-ranking-09 | Merged 512/2,048-triangle sources, all visited reuse states, corrected gates and equal-state loss; 8,192 updates | All support gates passed: 715 informative states. Every seed improved UV-off retention by 2.3–21.1%, but UV-on retention worsened 16.2–20.6%. No readiness claim. |
| policy-ranking-10 | Same verified data and objective; stop at 512 updates | Seeds 101/211 improved both modes (about 6% UV-on, 16–21% UV-off). Seed 307 regressed 24.6% UV-on, despite improving UV-off. Inconsistent; gate remains closed. |
| policy-ranking-11 | Keep one eighth of each experiment-09 ranking adjustment | Complete and negative: every UV-on comparison worsened; UV-off ranged from about +3.2% to -5.0%. Damping did not stabilize the learned policy. Raw optimizer checkpoints remain separate from blended inference models. |
| policy-ranking-12 | Same 512-update recipe as experiment 10, with UV-on teacher states only | UV-on improvement ranged from +4.6% to -19.4%, with per-LOD regressions in every seed. UV-off improved 10–12% despite its labels being absent. Removing UV-off labels did not resolve instability; this dataset is diagnostic only and cannot satisfy readiness support. |

The initializer is a migrated pretrained model, not a random network. Its UV
input column starts at zero; freezing the trunk prevents learning new dependence
on that explicit condition. Existing topology features can still differ between
UV modes. This is a capacity limitation, not proof of the cause of a regression.

Another mismatch is established: runtime uses the shared ranking head for both
predicted placements and endpoint reuse, while the original aligned teacher
audited predicted placements only. Single-action labels also do not guarantee that a
batch of actions is jointly safe. Complete runtime audits remain mandatory.

The policy-mixed experiment keeps the pretrained actor and all other training
and evaluation settings fixed. It exposes globally high-scoring actions but
does not correct execution-mode, target-condition or predecessor mismatches.
Negative experiments are retained.

The runtime observer now labels endpoint-reuse actions separately, using the
actual fixed request condition and preceding emitted LOD. Borrowed device trials
avoid downloading candidate meshes. Native tests cover both UV modes, batches
1 and 16, cancellation-poll parity and recovery after an observer exception.
Observation preserves runtime output and action budgets. Dedicated teacher
audits have separate statistics and bounded optional raster caching. Cached and
uncached real-mesh checks in both UV modes produced byte-identical action data,
geometry masks, requests and measured trajectories. Raster draws fell from
8,460 to 3,177 (UV on) and 7,620 to 2,847 (UV off); shared-workstation teacher
timings fell about 25%. Cache scope ends before the next borrowed trial, and
candidate revisions remain monotonic across observation and native execution.

Experiment 06 exposes insufficient teaching signal on dense inputs: observing
the first 16 states under a 64-trial budget barely reduces a 109k/150k-triangle
source. Most single actions have exactly equal global pixel/area measurements.
They remain honest ties; balancing categories cannot create missing information.

Source-density experiments use explicit classical-reduced training source
variants requesting 512 and 2,048 triangles from recorded parent assets.
Each variant is its own exact FP32 LOD0, with parent identity, requested/actual
counts, reducer configuration and source hash recorded. These are input-density
augmentations, not claims that the variants satisfy the parent's LOD audit.
Variants retain the parent's sampling identity and do not inflate asset counts.
Evaluation meshes, settings, budgets, initializer and three learner seeds stay
fixed. No new network architecture is assumed.

The first rebuilt source variants exposed an attribute-topology mismatch. The
coupled classical reducer retains original attribute wedges unless merging is
requested: the 512-triangle rock had 273 geometric vertices but 1,511 attribute
vertices, with 2–9 copies per position. It had no nonmanifold geometric edges.
The native edit representation supports at most two source wedges, so this
variant had no legal endpoint actions. Treating it as a CUDA or optimizer failure
would be incorrect. Complete zero-action observations are now retained as
ineligible data, rather than classified as incomplete measurements.

The next source preparation uses the existing reducer's `merge_wedges` option,
which fits compatible chart attributes while retaining seam boundaries. Its
algorithm identifier and exact source hash distinguish these variants from the
earlier ones. This changes only the training inputs; native action legality and
the two-wedge representation remain unchanged. That representation is still a
limitation for meshes with more complicated attribute junctions.

## Confirmed additional repairs

The action executor originally committed against full audit views only, while
outer generation also required distinct search views. Experiment 06's seed-101
UV-on shelves proposal reached 296 triangles internally, but generation emitted
523 after source-search rejection. Both gate sets now come from the generator
and constrain every committed batch. Endpoint teacher v2 (semantic version 9)
records both sets and uses their joint feasibility. A deterministic plane fixture
demonstrates the failure: an edge-on audit camera accepts boundary damage seen
by search cameras. The guarded generator must retain useful reductions with
zero outer visual-gate rejections.

Rank-only loss v2 normalizes preference pairs within each sampled state before
averaging states. Previously, bolt-cutters states carried about 3.6 times the
ranking mass of onion/boulder states despite category-balanced sampling. Tests
vary pool sizes and tied-positive counts and require equal per-state gradient
mass. Checkpoints reject a silent switch from the old loss.

Runtime teaching now distributes observations across each actual trajectory,
including its last visited state, instead of taking only the first states. A
count-only baseline supplies the schedule; exact mesh/error/action/iteration
parity remains mandatory. Support reports retain absent categories and UV modes.
The next learner is admitted only after each parent/UV mode has at least 16
informative states from at least two source scales.

Experiment 09 passes that admission check. Its exact sampler probabilities give
UV-on 49.8%, 47.4%, 52.3% and 64.3% of each manufactured, organic, rock and stress
parent's mass, respectively. The weakest parent/mode has 20 informative states.
This supports further controlled optimization, but does not establish that the
teacher's one-action preference predicts the eventual chain.

The shared ranking row remains an objective-coverage limitation: this runtime
teacher labels endpoint reuse only, while automatic output also ranks rebuilt
placements with the same row. The first rock LOD nevertheless regresses during
reuse itself (seed 101: 1,862 to 2,298 triangles after experiment 09), so missing
rebuild supervision cannot explain the whole failure. Geometry mask, frozen
optimizer and per-state hinge contracts passed a fifth Prophet review and
executable local checks; further mode-specific teaching remains outstanding.

The long experiment-09 runs moved the ranking row by about 64.7% of its initial
L2 norm. The 512-update runs moved it by about 7.9%. The next diagnostic fixes a
one-eighth interpolation of the long-run ranking adjustment, approximately
8.1% of the initial norm, across all three seeds. It tests conservative policy
adaptation without changing the actor, architecture or evaluation budget. This
is an inference artifact derived from two named models, not a resumable blended
optimizer checkpoint. The helper rejects any change to frozen parameters and
records both model hashes and the fraction in its provenance.

The reserved 12-asset pilot shares the `namaqualand_boulder` source family with
the old training boulder. Those old checkpoints are therefore ineligible for an
independent family-level pilot claim. The new aligned recipe uses `mountainside`
as its training rock and rejects family overlap with both the diagnostic and
reserved pilot before collecting data. The frozen pilot remains unchanged.

`policy-ranking-aligned.json` combines these confirmed contract/data repairs,
source augmentation and the equal-state learner. Initial and trained evaluation
within that experiment share the corrected runtime; any improvement there can
be attributed to learning rather than comparing different executors.

## Escalation gate

First require repeatable triangle improvements on the small development
diagnostic in both UV modes. Then use all 12 families of the frozen development
pilot, disjoint from the teacher meshes, across three learner seeds. Before any
experiment-13 checkpoint existed, `sequential-pilot-gate.json` froze a separate
small-source scenario: deterministic classical coupled Rebuild with wedge
merging, capped at 2,048 triangles, retaining smaller parents unchanged. Exact
FP32 derived sources, parent file hashes, preparation method, binary hashes and
actual source counts are recorded. These derived meshes are the new LOD0;
there is no claim of audited fidelity to the parents. Preparation failures
remain failures and no family can be removed after seeing results.

The original 12 source meshes and their manifest remain frozen for later scale
evaluation. The reason for a separate scenario is work headroom: one proposal
can remove at most 2,048 faces in 1,024 singleton trials, irrespective of network
quality. Alternative proposals are not concatenated. A full 150,000-face input
therefore needs a much larger trial budget for a useful reduction comparison.
This cap can conceal learning; it does not make the category-balanced aggregate
mathematically impossible in every case. The small-source scenario tests
learning feasibility across every family without pretending to validate scale.

Keep the gate of at least 5% relative improvement in category-balanced retained
triangles for every seed and UV mode, with no per-LOD triangle regression above
2%, complete valid audits and unchanged constant controls. Report all rows and
trial-budget terminations. This gate justifies a larger remote experiment; it
does not establish release quality, normal preservation or all-view fidelity.

Current status: **not remote-ready**. The optimizer learns the supplied labels,
but useful independent-mesh quality has not yet been made repeatable.
