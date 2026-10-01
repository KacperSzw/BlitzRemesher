# Appearance-preserving chain experiment

The frozen reference is graph v3 at commit ab6a8bd. Publish it unchanged before
this experiment. Keep the eight development assets, eight levels at 512→16 px,
3 px source cap, 2 px transitions, 50% changed area, original appearance weights,
20% added packed vertex cap, zero triangle overhead, and one graph pass.

Success requires 10% lower category-balanced chain retention versus the v3
appearance graph, no asset chain-total regression, and 25% fewer final-level
triangles on at least four assets spanning all four categories. Every delivered
reduced level must pass independent source and adjacent appearance audits.
Three paired repeats must each fit 4x the original appearance baseline. A pilot
signal precedes validation20, which requires 10% gain, no asset regression and
independent qualification. Held-out assets are reserved for release evidence.

Implement and compare separate stages: conservative coverage-only screening;
appearance-aware collapse ordering with unchanged emission; wedge attribute
fitting at existing positions; joint position/attribute fitting. Keep defaults
off. Stop adding complexity only when the success gates pass. Otherwise preserve
all negative results and do not promote. Use double contiguous coefficients for
active normal/RGB channels, discrete materials, retained discontinuities, and
existing topology/UV checks. Reuse preserves source bytes. No C ABI change.

The screening diagnostic compares fixed candidate bytes at 2/4/8/16/32 samples
on identical cameras, then different camera sets separately. A coarse screen
can reject only a coverage lower bound above the limit; uncertain candidates
receive the unchanged full audit. This does not change final sampling semantics.
Fixtures precede stool/trunk smoke runs and the complete pilot. Use four workers,
24 GiB combined memory and checkpoint each batch by 50 minutes. Independent
qualification uses a fresh 0xA1172027 rotation, 642+64 cameras, 8x refined to 32x.
Resource limits and cancellations are incomplete, never passing decisions.

Primary references: https://hhoppe.com/newqem.pdf and
https://hhoppe.com/proj/minqem/. Their appearance quadrics are proposal costs,
not guarantees about the sampled screen-space maximum metric.
