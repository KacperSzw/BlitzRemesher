// A bounded two-asset before/after diagnostic, separate from full quality gates.
import fs from 'node:fs';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { createHash } from 'node:crypto';
import { read, write } from './artifacts.mjs';
import { boundedProcess } from './bounded-process.mjs';
import { compareAudits, auditTermination, successful } from './quality-metrics.mjs';

const hash = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest = 'research/neural/action-diagnostic.json';
const settings = 'research/neural/teacher-capability-quality.json';
const binary = 'build/neural/blitz-neural-diagnostics';

export async function runCapabilityQuality({
  directory,
  initialModel,
  finalModel,
  deadline,
  signal,
  execute = boundedProcess,
  now = Date.now,
}) {
  if (fs.existsSync(directory)) throw new Error('choose a fresh capability quality directory');
  fs.mkdirSync(directory, { recursive: true });
  const started = now();
  const report = {
    version: 1,
    scenario: 'two-development-asset-four-LOD-capability',
    complete: false,
    score_eligible: false,
    score: null,
    quality_proven: false,
    strategy_adopted: false,
    training_started: false,
    started,
    runs: [],
    limitations: [
      'Reduced diagnostic camera and LOD schedule; not the full development quality protocol.',
      'Constant ranking means equal edge scores with stable action order; v4 placement predictions still depend on the model.',
      'Best observed comparator counts are pointwise references, not an independent classical baseline or a realizable combined chain.',
      'Trial-limited results are matched-budget observations, not proof that reduction is exhausted.',
    ],
  };
  const persist = () => write(path.join(directory, 'report.json'), report);
  try {
    if (!Number.isFinite(deadline) || deadline <= started)
      throw new Error('invalid capability quality deadline');
    const end = Math.min(deadline, started + 20 * 60000);
    const config = read(settings);
    const assets = read(manifest).assets;
    if (
      assets.length !== 2 ||
      assets.some((asset) => asset.split !== 'development') ||
      JSON.stringify(config.audit_rankings) !== JSON.stringify(['constant', 'learned'])
    )
      throw new Error('invalid two-asset capability diagnostic contract');
    report.contract = {
      manifest_sha256: hash(manifest),
      settings_sha256: hash(settings),
      binary_sha256: hash(binary),
      models: { initial: hash(initialModel), final: hash(finalModel) },
      settings: config,
      assets: assets.map(({ id, category, triangles }) => ({ id, category, triangles })),
      maximum_ms: end - started,
      per_model_maximum_ms: 10 * 60000,
    };
    persist();
    for (const [variant, model] of [
      ['initial', initialModel],
      ['final', finalModel],
    ]) {
      const grace = 3000;
      const maximum = Math.min(10 * 60000, end - now()) - grace;
      if (signal?.aborted || maximum <= 0)
        throw new Error('capability quality deadline/cancellation');
      const output = path.join(directory, variant + '.json');
      const log = path.join(directory, variant + '.log');
      const row = { variant, model_sha256: report.contract.models[variant], output, log };
      report.runs.push(row);
      const fd = fs.openSync(log, 'wx');
      try {
        row.process = await execute(binary, ['--audit-model', manifest, model, settings, output], {
          maximum,
          grace,
          signal,
          stdio: ['ignore', fd, fd],
        });
      } finally {
        fs.closeSync(fd);
        if (fs.existsSync(output)) {
          row.output_sha256 = hash(output);
          try {
            row.audit = read(output);
          } catch (error) {
            row.output_error = String(error);
          }
        }
        persist();
      }
      if (!successful(row.process)) throw new Error(variant + ' capability audit did not finish');
      for (const key of ['manifest_sha256', 'settings_sha256', 'binary_sha256'])
        if (row.audit?.[key] !== report.contract[key])
          throw new Error('capability audit identity mismatch: ' + key);
      if (
        row.audit.model_sha256 !== row.model_sha256 ||
        hash(model) !== row.model_sha256 ||
        hash(manifest) !== report.contract.manifest_sha256 ||
        hash(settings) !== report.contract.settings_sha256 ||
        hash(binary) !== report.contract.binary_sha256
      )
        throw new Error('capability audit inputs changed');
      if (
        !isDeepStrictEqual(
          row.audit.expected_assets,
          assets.map(({ id, category }) => ({ id, category })),
        ) ||
        row.audit.rows.some((asset) => asset.lods?.length !== config.levels)
      )
        throw new Error('capability audit asset or LOD schedule differs');
      row.termination = auditTermination(row.audit);
      persist();
    }
    const [initial, final] = report.runs.map((run) => run.audit);
    report.learned_comparison = compareAudits(initial, final);
    report.constant_ranking_comparison = compareAudits(initial, final, 'constant');
    report.uncensored = report.runs.every((run) => run.termination.uncensored);
    report.best_observed_comparators = final.rows
      .filter((row) => row.ranking === 'learned')
      .map((row) => {
        const references = [
          ...initial.rows
            .filter((r) => r.asset === row.asset)
            .map((r) => ({ method: 'initial-' + r.ranking, row: r })),
          {
            method: 'final-constant',
            row: final.rows.find((r) => r.asset === row.asset && r.ranking === 'constant'),
          },
        ];
        return {
          asset: row.asset,
          lods: row.lods.slice(1).map((lod, index) => {
            const triangles = Math.min(...references.map((r) => r.row.lods[index + 1].triangles));
            return {
              level: index + 1,
              triangles,
              methods: references
                .filter((r) => r.row.lods[index + 1].triangles === triangles)
                .map((r) => r.method),
              final_learned_triangles: lod.triangles,
              final_delta: lod.triangles - triangles,
            };
          }),
        };
      });
    report.complete = true;
  } catch (error) {
    report.error = String(error);
  } finally {
    report.finished = now();
    persist();
  }
  return report;
}
