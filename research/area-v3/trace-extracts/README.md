# One-asset proposal traces

These three diagnostic replays use the final rebuilt executable and set
`research.trace=true` on the frozen cap-0.5 pilot settings. The corresponding
one-asset manifests and trace configs are included here. Each run directory
contains its metadata, row, and exported mesh. The metadata records its binary,
protocol, config, manifest, and input hashes.

| Trace run | Scored pilot row | Output, attribute, and canonical-input hashes |
|---|---|---|
| `shelves-fixed` | cap-0.5 / painted wooden shelves | all match |
| `shelves-adaptive` | cap-0.5-adaptive / painted wooden shelves | all match |
| `hatch-quadric` | cap-0.5 / Hatch, Crew, Apollo 11 | all match |

Inspect `result.proposals` in each row for requested and achieved triangle
counts, the direct/progressive origin, the first failed gate, and reducer
rejection counters. These replays diagnose the scored outputs; their one-asset
SCORE values are not comparisons with the eight-asset pilot.
