# Rebuilt-storage comparison

Open [index.html](index.html) directly in a browser. It embeds the geometry and
works offline. The eight asset rows stay fixed while switching methods, LODs,
display modes and configured worst-view cameras. Settings and independent
qualification counts appear above the meshes.

All six methods use the recorded appearance scenario: eight LODs, 512→16 px,
3 px source cap, 2 px transition cap, 50% changed area, 20% added vertex storage.
The method selector changes recorded runs; it does not start a new bake.

**The new shared-source run has zero independently qualified reduced chains.**
Its configured-view SCORE improves modestly, but all five reduced chains fail
at least one dense independent comparison. The other three remain exact source
fallbacks. Keep that distinction when reading the curves or inspecting exports.
Earlier screen/ordering Bell exports retain their archived passing audits.

[Decision and failures](../../research/density/DECISION.md) ·
[measurements](../../research/density/REPORT.md) ·
[progress chart](../../research/density/progress.svg) ·
[verification](../../research/density/VERIFICATION.md) ·
[reproduction](../../research/density/REPRODUCE.md).

Raw glTF, buffer and engine-manifest links address local generated run outputs;
they are not embedded downloads or qualified delivery recommendations. The
standalone board and its geometry remain usable after cloning without those
outputs. The library exporter shares actual vertex buffers and retains material
IDs; engines still supply the original material payload. Textures, normal maps,
transparency and skinning are outside the audit contract.
