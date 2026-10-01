import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { runCapabilityQuality } from '../scripts/neural/teacher-capability-quality.mjs';

const repository = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const originalCwd = process.cwd();
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-capability-quality-'));
for (const file of [
  'research/neural/action-diagnostic.json',
  'research/neural/teacher-capability-quality.json',
]) {
  fs.mkdirSync(path.dirname(path.join(fixture, file)), { recursive: true });
  fs.copyFileSync(path.join(repository, file), path.join(fixture, file));
}
fs.mkdirSync(path.join(fixture, 'build/neural'), { recursive: true });
fs.writeFileSync(path.join(fixture, 'build/neural/blitz-neural-diagnostics'), 'test-owned-binary');
fs.writeFileSync(path.join(fixture, 'initial.blzn'), 'initial-model');
fs.writeFileSync(path.join(fixture, 'final.blzn'), 'final-model');
process.chdir(fixture);
after(() => {
  process.chdir(originalCwd);
  fs.rmSync(fixture, { recursive: true, force: true });
});
const hash = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = (file) => JSON.parse(fs.readFileSync(file));
const success = { code: 0, signal: null, timed_out: false, cancelled: false, success: true };

function audit(command, args) {
  const [, manifest, model, settings] = args;
  const config = read(settings);
  const assets = read(manifest).assets;
  const measurement = {
    complete: true,
    passed: true,
    error: 0.25,
    error_kind: 'measured',
    resource_limited: false,
  };
  return {
    version: 2,
    complete: true,
    score: null,
    binary_sha256: hash(command),
    manifest_sha256: hash(manifest),
    settings_sha256: hash(settings),
    model_sha256: hash(model),
    visual_settings_sha256: 'same-visual-settings',
    execution_settings_sha256: 'same-work-budget',
    execution_settings: { audit_rankings: config.audit_rankings },
    expected_assets: assets.map(({ id, category }) => ({ id, category })),
    rows: assets.flatMap((asset) =>
      config.audit_rankings.map((ranking) => ({
        asset: asset.id,
        category: asset.category,
        ranking,
        status: 'complete',
        seconds: 1,
        neural: {
          action_diagnostics_version: 1,
          action_proposals: [{ stop_reason: 'trial_budget' }],
        },
        lods: Array.from({ length: config.levels }, (_, level) => ({
          triangles: level
            ? 100 - level * 10 - (model === 'final.blzn' && ranking === 'learned' ? 5 : 0)
            : 100,
          source: measurement,
          adjacent: measurement,
        })),
      })),
    ),
  };
}
let next = 0;
async function run(execute, overrides = {}) {
  return runCapabilityQuality({
    directory: path.join(fixture, 'case-' + next++),
    initialModel: 'initial.blzn',
    finalModel: 'final.blzn',
    deadline: 1200000,
    now: () => 0,
    execute,
    ...overrides,
  });
}
function writeAudit(command, args, change = () => {}) {
  const result = audit(command, args);
  change(result);
  fs.writeFileSync(args.at(-1), JSON.stringify(result));
}

test('matched before/after rows retain trial censorship, measured errors and per-LOD comparators', async () => {
  const calls = [];
  const report = await run(async (command, args, options) => {
    calls.push(options);
    writeAudit(command, args);
    return success;
  });
  assert.equal(report.complete, true);
  assert.equal(report.quality_proven, false);
  assert.equal(report.score, null);
  assert.equal(report.strategy_adopted, false);
  assert.equal(report.uncensored, false);
  assert.equal(report.runs[0].termination.reasons.trial_budget, 4);
  assert.equal(report.runs[0].audit.rows[0].lods[1].source.error, 0.25);
  assert(
    report.learned_comparison.per_lod_deltas.every((asset) =>
      asset.triangle_deltas.every((delta) => delta === -5),
    ),
  );
  assert(
    report.best_observed_comparators.every((asset) =>
      asset.lods.every((lod) => lod.final_delta === -5),
    ),
  );
  assert(calls.every((options) => options.maximum + options.grace <= 10 * 60000));
});

test('incomplete or mismatched audits cannot become a capability comparison', async () => {
  for (const change of [
    (a) => (a.complete = false),
    (a) => (a.rows[0].lods[1].source.complete = false),
    (a) => (a.rows[0].neural.confirmation_cancelled = 1),
    (a) => (a.binary_sha256 = 'wrong-binary'),
    (a) => (a.model_sha256 = 'wrong-model'),
    (a) => a.rows.pop(),
    (a) => a.rows[0].lods.pop(),
  ]) {
    const result = await run(async (command, args) => {
      writeAudit(command, args, change);
      return success;
    });
    assert.equal(result.complete, false);
    assert.equal(result.score, null);
    assert(result.error);
    assert.equal(result.runs.length, 1);
    assert(result.runs[0].audit); // Failure remains inspectable.
  }
});

test('signal failure preserves partial native output and never launches the final audit', async () => {
  let calls = 0;
  const result = await run(async (command, args) => {
    ++calls;
    writeAudit(command, args, (a) => {
      a.complete = false;
      a.rows.pop();
    });
    return { code: null, signal: 'SIGTERM', timed_out: true, cancelled: false, success: false };
  });
  assert.equal(calls, 1);
  assert.equal(result.complete, false);
  assert.equal(result.runs[0].process.signal, 'SIGTERM');
  assert.equal(result.runs[0].audit.rows.length, 3);
  assert.equal(result.runs[0].output_sha256, hash(result.runs[0].output));
});

test('short outer deadlines reserve termination time and prevent another audit after expiry', async () => {
  let clock = 0,
    calls = 0;
  const result = await run(
    async (command, args, options) => {
      ++calls;
      assert(options.maximum + options.grace <= 7000);
      writeAudit(command, args);
      clock = 7000;
      return success;
    },
    { deadline: 7000, now: () => clock },
  );
  assert.equal(calls, 1);
  assert.equal(result.complete, false);
  assert.match(result.error, /deadline/);
  const controller = new AbortController();
  controller.abort();
  const stopped = await run(
    async () => {
      throw new Error('must not execute');
    },
    { signal: controller.signal },
  );
  assert.equal(stopped.runs.length, 0);
  assert.equal(stopped.complete, false);
});
