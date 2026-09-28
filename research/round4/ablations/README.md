# Replaying the Round 4 ablations

The archived runs retain their original binary, build, source-tree, input,
configuration and camera hashes. Frozen executables and their matching
build stamps are also retained locally under build/round4-{baseline,relaxed,
link,search}; these ignored files are not distributed in Git.

The patches here restore the old reducer/search behavior on top of the
accepted implementation while retaining the newer export and diagnostic
interfaces. Use a separate checkout or worktree so ongoing experiments are
unaffected. The source corpus must be available at its manifest paths.

| Variant | Patches to apply | Objective in configuration |
|---|---|---|
| Previous implementation | legacy-ownership.patch, legacy-topology.patch, legacy-search.patch | quadric |
| Relaxed locks | legacy-ownership.patch, legacy-search.patch | topology_relaxed |
| Neighbor guard alone | legacy-ownership.patch, legacy-search.patch | quadric |
| Guard + improved search prototype | legacy-ownership.patch | quadric |
| Final implementation, including progressive ownership fix | none | quadric |

For example, from the separate repository root:

    git apply research/round4/ablations/legacy-ownership.patch research/round4/ablations/legacy-search.patch
    nix develop path:. --command cmake --build build/release -j 2
    bash tools/stamp-build.sh
    build/release/blitz bench research/pilot.json research/round4/configs/s512-cap8.json NEW_OUTPUT

Use fresh output directories. The same source/configuration/protocol must
remain frozen across paired comparisons. New builds have new binary/build
hashes and may contain additional diagnostic/export metadata; these patches
reproduce proposal behavior, not historical executable bytes. Exact-runtime
export compaction leaves all scheduled slots and their score unchanged.

The native baseline preceding these changes is Git revision 77adf6f.
Raw reducer minima are unaudited diagnostics, not accepted visual results.
Do not rank different starting sizes, source caps or level counts as if they
were algorithm improvements under one contract.
