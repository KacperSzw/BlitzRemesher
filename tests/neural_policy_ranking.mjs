import { test } from 'node:test';
import assert from 'node:assert/strict';
import { comparePolicyRanking } from '../scripts/neural/policy-ranking-quality.mjs';
import { rankingSupport } from '../scripts/neural/policy-ranking-support.mjs';

test('teacher support retains absent UV modes and reports sampler-weighted pair imbalance', () => {
  const shard = (asset, category, preserve_uv, n, preferred) => ({
    index: { asset, category, source_triangles: n },
    contract: { preserve_uv },
    trajectory: [
      {
        triangles: n,
        preferred_mask: preferred,
        rows: Array.from({ length: 16 }, () => ({ known_mask: 24 })),
      },
    ],
  });
  const a = shard('a', 'rocks', false, 300, 1);
  const tied = shard('a', 'rocks', true, 900, 65535);
  const b = shard('b', 'organic', false, 500, 255);
  const result = rankingSupport([a, tied, b], 1, 1);
  assert.equal(result.admitted, false);
  assert.deepEqual(
    result.rows.map((r) => [r.asset, r.preserve_uv, r.states, r.informative]),
    [
      ['a', false, 1, 1],
      ['a', true, 1, 0],
      ['b', false, 1, 1],
      ['b', true, 0, 0],
    ],
  );
  assert.deepEqual(result.category_pair_weight, [
    { category: 'rocks', expected_pairs_per_sampled_state: 15 },
    { category: 'organic', expected_pairs_per_sampled_state: 64 },
  ]);
  assert.equal(result.largest_pair_weight_ratio, 64 / 15);
  assert.equal(rankingSupport([a, shard('a', 'rocks', true, 900, 1)], 1, 1).admitted, true);
  assert.equal(rankingSupport([a, shard('a', 'rocks', true, 900, 1)], 1, 2).admitted, false);
});

function audit(triangles = [80, 60], limit = 2.7, area = 0.4) {
  const measurement = {
    complete: true,
    passed: true,
    error_px: limit / 2,
    coverage_upper_px: limit / 2,
    changed_area: area / 2,
  };
  return {
    version: 2,
    complete: true,
    manifest_sha256: 'meshes',
    visual_settings_sha256: 'coverage',
    execution_settings_sha256: 'budget',
    execution_settings: { audit_rankings: ['constant', 'learned'] },
    visual_settings: { profile: 'coverage', max_changed_area: area },
    expected_assets: [{ id: 'a', category: 'rocks' }],
    rows: ['constant', 'learned'].map((ranking) => ({
      asset: 'a',
      category: 'rocks',
      ranking,
      status: 'complete',
      seconds: 1,
      neural: {
        action_diagnostics_version: 1,
        inference_seconds: 0.1,
        action_proposals: [{ stop_reason: 'trial_budget' }],
      },
      lods: [
        { triangles: 100 },
        ...(ranking === 'constant' ? [90, 80] : triangles).map((count) => ({
          triangles: count,
          source_limit: limit,
          transition_limit: limit,
          source: { ...measurement },
          adjacent: { ...measurement },
        })),
      ],
    })),
  };
}

test('ranking comparison preserves per-LOD regressions and trial caps despite chain improvement', () => {
  const result = comparePolicyRanking(audit(), audit([81, 40]));
  assert.equal(result.constant_control_unchanged, true);
  assert.ok(result.relative_retained_improvement > 0);
  assert.deepEqual(result.per_lod_regressions, [{ asset: 'a', level: 1, triangles: 1 }]);
  assert.equal(result.worst_relative_lod_increase, 81 / 80 - 1);
  assert.equal(result.termination.after.uncensored, false);
  assert.equal(result.score, null);
});

test('frozen actor control ignores timing noise but rejects geometry and audit changes', () => {
  const candidate = audit([70, 40]);
  candidate.rows[0].seconds = 100;
  candidate.rows[0].neural.inference_seconds = 20;
  assert.equal(comparePolicyRanking(audit(), candidate).constant_control_unchanged, true);
  for (const change of [
    (row) => row.lods[1].triangles--,
    (row) => (row.lods[1].source.error_px /= 2),
    (row) => (row.neural.action_proposals[0].stop_reason = 'no_accepted_action'),
  ]) {
    const different = structuredClone(candidate);
    change(different.rows[0]);
    assert.throws(() => comparePolicyRanking(audit(), different), /control changed/);
  }
});

test('both audits require finite measurements within test-owned pixel and area limits', () => {
  for (const [limit, area] of [
    [1.3, 0.2],
    [3.1, 0.7],
  ]) {
    const before = audit([80, 60], limit, area);
    assert.doesNotThrow(() => comparePolicyRanking(before, structuredClone(before)));
    for (const reference of ['source', 'adjacent'])
      for (const change of [
        (m) => (m.error_px = NaN),
        (m) => (m.error_px = -0.1),
        (m) => (m.coverage_upper_px = limit + 0.1),
        (m) => (m.changed_area = area + 0.1),
        (m) => (m.changed_area = null),
        (m) => (m.nonfinite_error = true),
        (m) => (m.resource_limited = true),
        (m) => (m.cancelled = true),
        (m) => (m.complete = false),
      ]) {
        const bad = structuredClone(before);
        change(bad.rows[1].lods[2][reference]);
        assert.throws(() => comparePolicyRanking(before, bad));
      }
  }
});
