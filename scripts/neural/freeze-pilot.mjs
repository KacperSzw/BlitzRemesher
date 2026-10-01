// Metadata-only selection. Never replace a failing asset after seeing quality.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
export function selectPilot(corpus, selection) {
  const selected = new Set(selection.assets.map((a) => a.id));
  const rows = [];
  for (const category of ['manufactured', 'organic', 'rocks', 'stress']) {
    const groups = new Set();
    const available = corpus.assets
      .filter(
        (a) =>
          selected.has(a.id) &&
          a.split === 'development' &&
          a.category === category &&
          a.triangles <= 150000,
      )
      .sort((a, b) => a.triangles - b.triangles || a.id.localeCompare(b.id, 'en'))
      .filter((a) => {
        if (groups.has(a.source_group)) return false;
        groups.add(a.source_group);
        return true;
      });
    if (available.length < 3)
      throw Error(`Need three distinct development source groups in ${category}`);
    for (const index of [0, Math.floor(available.length / 2), available.length - 1])
      rows.push(available[index]);
  }
  if (new Set(rows.map((a) => a.source_group)).size !== 12)
    throw Error('Pilot source groups overlap');
  return rows;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [directory] = process.argv.slice(2);
  if (!directory || fs.existsSync(directory)) throw Error('freeze-pilot.mjs FRESH_DIRECTORY');
  const files = ['research/corpus.json', 'research/neural/training-manifest.json'];
  const buffers = files.map((f) => fs.readFileSync(f)),
    [corpus, selection] = buffers.map((b) => JSON.parse(b));
  const assets = selectPilot(corpus, selection);
  const hashes = Object.fromEntries(
    files.map((f, i) => [f, createHash('sha256').update(buffers[i]).digest('hex')]),
  );
  fs.mkdirSync(directory, { recursive: true });
  const write = (name, value) =>
    fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n', {
      flag: 'wx',
    });
  write('selection.json', {
    version: 1,
    selection:
      'Frozen metadata-only pilot: three distinct source groups/category; <=150000 triangles; sorted (triangles,id), indices 0,floor(n/2),n-1.',
    source_hashes: hashes,
    score_eligible: false,
    assets,
  });
  write('curriculum.json', {
    version: 1,
    conditions: assets.flatMap((a) =>
      [32, 64, 128]
        .map((pixels) => ({ asset: a.id, pixels, previous_steps: 0 }))
        .concat({ asset: a.id, pixels: 128, previous_steps: 2 }),
    ),
  });
  console.log(
    JSON.stringify(
      assets.map(({ id, category, triangles }) => ({ id, category, triangles })),
      null,
      2,
    ),
  );
}
