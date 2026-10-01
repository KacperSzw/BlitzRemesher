import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateExpanded } from '../scripts/neural/corpus.mjs';
function fixture() {
  const asset = (id, split) => ({
    id,
    source_group: id,
    geometry_sha256: id,
    split,
    license: 'CC0-1.0',
    opaque: true,
    import_status: 'ok',
  });
  const a = asset('train', 'development'),
    b = asset('validation', 'validation'),
    c = asset('release', 'held_out'),
    d = asset('new', 'development');
  return {
    original: { assets: [a, b, c] },
    allowed: { assets: [a] },
    corpus: { assets: [a, b, c, d] },
    training: { assets: [a, d] },
    validation: { assets: [b] },
    trainingGroups: 2,
    validationGroups: 1,
  };
}
test('expanded selections preserve original assets and accept multiple valid group counts', () => {
  for (const n of [1, 2]) {
    const x = fixture();
    x.training.assets = x.training.assets.slice(0, n);
    x.trainingGroups = n;
    assert.equal(validateExpanded(x).training_groups, n);
  }
});
test('family leakage, copied geometry, wrong splits, changed originals and invalid sources fail', () => {
  const changes = [
    (x) => {
      x.corpus.assets[3].source_group = 'release';
    },
    (x) => {
      x.corpus.assets[3].source_group = 'validation';
    },
    (x) => {
      x.corpus.assets[3].geometry_sha256 = 'train';
    },
    (x) => {
      x.corpus.assets[3].split = 'held_out';
    },
    (x) => {
      x.corpus.assets[3].opaque = false;
    },
    (x) => {
      x.corpus.assets[3].license = 'unknown';
    },
    (x) => {
      x.corpus.assets[0] = { ...x.corpus.assets[0], changed: true };
    },
    (x) => {
      x.corpus.assets.push(x.corpus.assets[0]);
    },
  ];
  for (const change of changes) {
    const x = fixture();
    change(x);
    assert.throws(() => validateExpanded(x));
  }
});
