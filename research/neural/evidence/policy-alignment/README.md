# Sequential endpoint learning evidence

The current decision and next steps are in [the handoff](../../HANDOFF.md) and
[investigation](../../POLICY-ALIGNMENT.md). This archive has no aggregate quality
score and does not mark the model ready for remote training.

`manifest.json` records each gzip file's original byte count and SHA-256. All
836 entries were decompressed and verified before committing. Negative runs,
interrupted audits, the invalid bitmap-parser probe and the corrected probe are
retained. To verify the archive independently, from the repository root:

```sh
node --input-type=module <<'JS'
import fs from 'node:fs';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
const root = 'research/neural/evidence/policy-alignment';
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json')));
for (const entry of manifest.entries) {
  const bytes = gunzipSync(fs.readFileSync(path.join(root, entry.path)));
  const hash = createHash('sha256').update(bytes).digest('hex');
  if (bytes.length !== entry.bytes || hash !== entry.sha256)
    throw Error('Evidence mismatch: ' + entry.path);
}
console.log('Verified ' + manifest.entries.length + ' entries');
JS
```

The full endpoint scorer fits the training labels, but its ordinary UV-on
seed-101 rollout regresses on the diagnostic rock. Restricting its choices to
the frozen collector's native 16-action teacher pool recovers that result:

| UV-on rock | LOD1 | LOD2 | LOD3 |
| --- | ---: | ---: | ---: |
| Ordinary initializer | 1,638 | 820 | 410 |
| Ordinary trained scorer | 1,936 | 968 | 484 |
| Supported initializer | 1,638 | 820 | 410 |
| Supported trained scorer | 1,256 | 628 | 314 |

Both supported runs pass exact final source and adjacent visual gates. Their
native binary, source, visual and execution hashes match. The trained global
top action lies outside support in 3,705/3,709 rescoring events; the initializer
has zero escapes. This is a conditional local learning signal on one seed,
one development rock and UV-on. The selector itself runs collector inference
and geometric rankings. Both UV modes and the frozen full pilot remain required
before paid training; the ordinary neural runtime has not passed that gate.

Key directories:

- `endpoint-ranking-01`: all three training health reports, the failed ordinary
  output comparison, stopped follow-up and explicit partial-result decision.
- `endpoint-support-01` and `endpoint-support-initial-01`: exact diagnostic
  settings, hypotheses, native traces, bounded runner, result and matched control.
- `endpoint-diversity-01`: one new teaching shard's metadata and descriptive
  probe, including the retained invalid result and its correction.
- `endpoint-plane-probe-01`: both UV modes of the classical current-plane
  comparison; rock improves but some shelves levels regress.
- `next-step-reviews-01`: Prophet's implementation, quality, support and
  conditional remote-orchestration reviews, with the supplied code contexts.
- `final-verification`: passing CPU/native/sanitizer checks and the initial
  support implementation's failing observer test before its correction.

Model, optimizer and teaching-data blobs remain in the shared local run
directories listed in the handoff. They are not bundled here. Archived JSON
reports retain their hashes; a new machine must copy the exact blobs and input
assets before reproducing a comparison.
