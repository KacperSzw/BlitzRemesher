import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { widthDecision } from '../scripts/neural/width-quality.mjs';
import {
  auditMetrics,
  compareAudits,
  auditTermination,
} from '../scripts/neural/quality-metrics.mjs';
import { budgetSettings, compareBudgetRound } from '../scripts/neural/budget-quality.mjs';
const measurement = { complete: true, passed: true };
function audit(triangles = [80, 60], seconds = 1) {
  return {
    version: 2,
    complete: true,
    manifest_sha256: 'inputs',
    visual_settings_sha256: 'gates',
    execution_settings_sha256: 'work',
    execution_settings: { audit_rankings: ['learned'] },
    expected_assets: [
      { id: 'a', category: 'rocks' },
      { id: 'b', category: 'organic' },
    ],
    rows: ['a', 'b'].map((asset, i) => ({
      asset,
      category: i ? 'organic' : 'rocks',
      ranking: 'learned',
      status: 'complete',
      seconds,
      neural: {
        action_diagnostics_version: 1,
        action_proposals: [{ stop_reason: 'target_reached' }],
      },
      lods: [
        { triangles: 100 },
        ...triangles.map((count) => ({
          triangles: count,
          source: measurement,
          adjacent: measurement,
        })),
      ],
    })),
  };
}
const success = (audit) => ({ code: 0, audit });
test('width choice uses all LODs, all categories and every seed; time stays diagnostic', () => {
  const seeds = [7, 19, 37],
    rows = seeds.flatMap((seed) => [
      { seed, width: 64, ...success(audit()) },
      { seed, width: 128, ...success(audit([79, 60], 10)) },
    ]);
  assert.equal(widthDecision(rows, seeds).selected_width, 128); // Less than 5%, slower, still an audited improvement.
  const regression = structuredClone(rows);
  regression.at(-1).audit = audit([90, 60]);
  assert.equal(widthDecision(regression, seeds).selected_width, 64); // Same last LOD, worse chain.
  for (const change of [
    (a) => (a.complete = false),
    (a) => (a.visual_settings_sha256 = 'changed'),
    (a) => (a.execution_settings_sha256 = 'changed'),
    (a) => (a.rows[0].status = 'failed'),
    (a) => a.rows.pop(),
  ]) {
    const bad = structuredClone(rows);
    change(bad.at(-1).audit);
    assert.equal(widthDecision(bad, seeds).selected_width, 64);
  }
  assert.equal(widthDecision(rows.slice(0, -1), seeds).selected_width, 64);
  for (const status of [
    { code: null, signal: 'SIGTERM' },
    { code: 7 },
    { code: undefined },
    { timed_out: true },
    { cancelled: true },
  ]) {
    const bad = structuredClone(rows);
    Object.assign(bad[0], status);
    assert.equal(widthDecision(bad, seeds).passed, false);
  }
});
test('categories receive equal weight and per-LOD regressions remain visible', () => {
  const before = audit([80, 60]),
    after = audit([90, 40]);
  const extra = structuredClone(before.rows[0]);
  extra.asset = 'c';
  extra.lods[1].triangles = 100;
  extra.lods[2].triangles = 100;
  before.expected_assets.push({ id: 'c', category: 'rocks' });
  before.rows.push(extra);
  assert.equal(auditMetrics(before).retained_ratio, ((0.7 + 1) / 2 + 0.7) / 2);
  const comparison = compareAudits(audit([80, 60]), after);
  assert.ok(comparison.retained_ratio_improvement > 0);
  assert.deepEqual(comparison.per_lod_regressions, [
    { asset: 'a', level: 1, triangles: 10 },
    { asset: 'b', level: 1, triangles: 10 },
  ]);
});
test('metrics reject missing, duplicate, failed, unaudited and increasing LOD rows', () => {
  for (const mutate of [
    (a) => a.rows.pop(),
    (a) => a.rows.push(a.rows[0]),
    (a) => (a.rows[0].lods[2].source.passed = false),
    (a) => (a.rows[0].lods[2].triangles = 90),
    (a) => (a.rows[0].seconds = NaN),
    (a) => delete a.rows[0].category,
    (a) => (a.rows[0].neural.confirmation_cancelled = 1),
  ]) {
    const invalid = structuredClone(audit());
    mutate(invalid);
    assert.throws(() => auditMetrics(invalid));
  }
  assert.throws(() =>
    compareAudits(audit(), { ...audit(), execution_settings_sha256: 'extra work' }),
  );
});
test('trial caps and missing stop diagnostics never establish exhaustion', () => {
  const limited = audit();
  limited.rows[0].neural.action_proposals[0].stop_reason = 'trial_budget';
  assert.equal(auditTermination(limited).uncensored, false);
  assert.equal(
    compareBudgetRound([success(limited), success(audit())], ['learned']).uncensored,
    false,
  );
  const unknown = audit();
  delete unknown.rows[0].neural.action_diagnostics_version;
  assert.equal(auditTermination(unknown).known, false);
  const empty = audit();
  empty.rows[0].neural.action_proposals = [];
  assert.equal(auditTermination(empty).known, false);
  assert.equal(auditTermination(empty).uncensored, false);
  const infeasible = audit();
  infeasible.rows[0].neural.action_proposals[0].stop_reason = 'infeasible_seed';
  assert.equal(auditTermination(infeasible).known, true);
  assert.equal(auditTermination(infeasible).uncensored, false);
  for (const reason of ['no_legal_actions', 'no_accepted_action', 'target_reached']) {
    const complete = audit();
    complete.rows[0].neural.action_proposals[0].stop_reason = reason;
    assert.equal(auditTermination(complete).uncensored, true);
  }
});
test('budget plan changes only explicit execution settings and enforces bounded increasing work', () => {
  const plan = {
    version: 1,
    profile: 'coverage',
    action_batch: 7,
    trial_budgets: [3, 19],
    rankings: ['learned'],
    maximum_minutes: 2,
    stop_when_uncensored: true,
  };
  const visual = { levels: 4, max_lod0_delta_px: 2.7, max_changed_area: 0.3 };
  assert.deepEqual(budgetSettings(plan, visual, 19), {
    ...visual,
    profile: 'coverage',
    action_trials: 19,
    action_batch: 7,
    audit_rankings: ['learned'],
  });
  for (const change of [
    { trial_budgets: [3, 3] },
    { trial_budgets: [3, 2] },
    { maximum_minutes: 51 },
    { action_batch: 65 },
    { rankings: ['learned', 'learned'] },
  ])
    assert.throws(() => budgetSettings({ ...plan, ...change }, visual, 3));
});
test('bounded runner requires both completed model executions and advances past a cap', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-quality-budget-'));
  try {
    const binary = root + '/audit.mjs';
    fs.writeFileSync(
      binary,
      `#!${process.execPath}\nimport fs from 'node:fs';import {createHash} from 'node:crypto';const settings=JSON.parse(fs.readFileSync(process.argv[5]));const a=${JSON.stringify(audit())};a.model_sha256=createHash('sha256').update(fs.readFileSync(process.argv[4])).digest('hex');for(const r of a.rows)r.neural.action_proposals[0].stop_reason=settings.action_trials===3?'trial_budget':'no_accepted_action';a.execution_settings_sha256=String(settings.action_trials);fs.writeFileSync(process.argv[6],JSON.stringify(a));if(process.env.TEST_EXIT==='signal')process.kill(process.pid,'SIGTERM');else process.exit(Number(process.env.TEST_EXIT));\n`,
      { mode: 0o700 },
    );
    const write = (name, value) => fs.writeFileSync(root + '/' + name, JSON.stringify(value));
    write('manifest.json', {
      assets: [
        { id: 'a', split: 'development' },
        { id: 'b', split: 'development' },
      ],
    });
    write('visual.json', {});
    write('plan.json', {
      version: 1,
      manifest: root + '/manifest.json',
      settings: root + '/visual.json',
      profile: 'coverage',
      action_batch: 7,
      trial_budgets: [3, 19, 37],
      rankings: ['learned'],
      maximum_minutes: 1,
      stop_when_uncensored: true,
    });
    fs.writeFileSync(root + '/initial.blzn', 'initial');
    fs.writeFileSync(root + '/final.blzn', 'final');
    for (const status of ['0', '7', 'signal']) {
      const output = root + '/out-' + status,
        result = spawnSync(
          process.execPath,
          [
            fileURLToPath(new URL('../scripts/neural/budget-quality.mjs', import.meta.url)),
            root + '/plan.json',
            root + '/initial.blzn',
            root + '/final.blzn',
            output,
          ],
          {
            env: { ...process.env, BLITZ_AUDIT_BINARY: binary, TEST_EXIT: status },
            encoding: 'utf8',
            timeout: 10000,
          },
        );
      assert.equal(result.error, undefined);
      assert.equal(result.status, status === '0' ? 0 : 1, result.stderr);
      const report = JSON.parse(fs.readFileSync(output + '/report.json'));
      assert.equal(report.complete, status === '0');
      assert.equal(report.score, null);
      if (status === '0') {
        assert.equal(report.rounds.length, 2);
        assert.equal(report.status, 'uncensored');
        assert.equal(report.rounds[0].uncensored, false);
      } else assert.equal(report.rounds[0].complete, false);
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
