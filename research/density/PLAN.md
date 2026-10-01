# Indexed rebuilt-proposal experiment

Checkpoint: 636d139. The appearance experiments selected only source-vertex
chains. Test whether the triangle-soup request cap hides feasible indexed
rebuilt candidates. Preserve the previous evidence and all quality/storage gates.

The first regression uses a connected 18-triangle, 16-vertex plane under a
budget that permits its vertex storage. A provider returns that mesh only when
the request reaches its triangle count. The old clamp fails the regression
before algorithm changes. Repeat at multiple source sizes and caps, in the
initial and graph searches. Separately use seam-heavy outputs to require actual
storage rejection and feedback, and verify zero/tiny caps never admit added bytes.

Opt-in density targets estimate the first rebuilt request from the input's
referenced-vertex ratio. A transient bitset counts referenced IDs without copies
of streams. Later calls use the actual emitted triangles and packed vertex bytes.
An over-budget or stalled proposal causes a lower request; spare bytes can permit
a denser request. Successful candidates remain in the existing pool. Rejected
appearance does not establish monotonicity. Existing calls, not additional calls,
fund this search. Tail probes and progressive remaining budgets follow the same
policy. This is a heuristic, never an admission proof.

First compare bounded stool/trunk smoke runs against screen-only and appearance
ordering. Choose the simplest stage with a useful signal. Freeze its executable
and run the same development pilot, then fresh independent 642+64-camera audits
(8x to 32x, seed 0xA1172028) of every reduced source/adjacent level. No held-out
assets. Keep incomplete results unscored.

Matched contract: eight levels, 512 to 16 px, 3 px source cap, 2 px adjacent cap,
50% changed area, original appearance weights, 20% added vertex storage, zero
triangle overhead, eight proposals per level and one graph pass. Compare to
the previous screen/ordering runs and the v3 appearance graph. Aim for >=10%
lower category-balanced chain retention, no asset regression and >=25% fewer
tail triangles on four assets across all categories. Dense audit failures cannot
qualify even if their pilot score improves. Validation and three paired timing
repetitions (<=4x original appearance baseline) require a useful pilot first.

The first policy remains opt-in until those checks justify promotion. Record
failures and select the next action from observed candidate gates and storage,
not from aggregate SCORE alone. Use at most four workers / 24 GiB and checkpoint
every batch within 50 minutes.

## Follow-up after the v1 smoke

Density-only geometric, fitted and uncoupled probes provided no new accepted
rebuilt chain on stool/trunk. The observed coupled output approaches three
attribute vertices per surviving face, so the original vertex ratio ceases to
predict its storage. The existing attribute-fitting output retains original
wedges even where their continuous interiors have collapsed together.

The next ablation adds a bounded emission pass. Original corner fans connect
only across two-sided manifold edges with continuous attribute indices and the
same material. Original boundaries, seams, nonmanifold edges and seam endpoints
anchor their fans; no merge may join two anchors. Collapsed interior edges can
merge compatible fans, retaining separate alpha and tangent-handedness values.
Fit only live continuous normal/RGB/UV fields; keep boundary UVs fixed. Roll back
the whole emission change on a surviving UV sign flip or degeneracy. This is
conservative about chart boundaries and can leave too many vertices. It does
not certify appearance. The same budget and independent audits still apply.

The v2 smoke rejected every whole-emission merge on a local UV reversal. The v3
postpass therefore restores affected original fans and their exact attributes,
then checks all surviving UV faces again, for at most four checks. An unresolved
reversal still rolls back the entire postpass. It never relaxes UV acceptance.
After observing repeated slightly over-budget outputs, target feedback leaves
1/16 of its estimated triangle capacity as proposal headroom. Actual bytes still
decide admission. Frozen source archives retain both negative intermediate builds.

## Source-sharing rebuilt buffers

The complete v3 pilot exported the same chains as appearance-screen: the storage
improvement did not produce an appearance-qualified rebuilt chain. A complete
rebuilt buffer must fit below 20% of source vertex storage even for a moderate
triangle reduction. This makes source sharing the next bounded storage ablation.

Experimental shared_rebuild matches complete vertex tuples byte for byte against
the original source, including normals, UV, RGBA and tangent handedness. Only
unmatched vertices consume the added-vertex budget. Preserve unchanged/endpoint
world coordinates exactly to avoid normalization round trips defeating sharing.
Each candidate owns one immutable contiguous pool; the selected chain combines
its source prefix and changed vertices into one shared allocation. Source IDs and
bytes stay unchanged, index buffers remain per runtime mesh, and glTF accessors
share the actual underlying vertex buffer. Keep the C ABI at version 5.

Initial regression: a planar six-triangle mesh contracts to four triangles with
four exact source vertices and one changed vertex. A 20% cap admits the changed
vertex and rejects a full rebuilt buffer. Test every attribute in the sharing
key, strided source storage, ownership after copying/destroying results, shared
pool accounting, and actual exported buffer/accessor sizes. Repeat stool/trunk
smokes before the fixed pilot. All existing quality limits and independent audit
requirements remain in force. The setting stays opt-in pending evidence.

The v4 pilot provides a modest configured-view gain but its trunk fails the
fresh audit at LOD1–6. The v5 correction retains graph paths with different
previously allocated pools: after paths meet at the same geometry, those pools
can make a later index mesh affordable. A controlled four-level fixture uses
two one-vertex pools, converges at a source-only mesh, and requires the second
pool again. Both tested caps admit one pool and reject two. Recheck the frozen
pilot after this correction; preserve the v4 result separately.
