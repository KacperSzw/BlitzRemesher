// Run audits only. This command never starts learning or provisions a GPU.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { boundedProcess } from './bounded-process.mjs';
import { auditTermination, compareAudits, successful } from './quality-metrics.mjs';
import { read, write } from './artifacts.mjs';
const digest = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
export function budgetSettings(plan, visual, budget) {
  if (
    plan.version !== 1 ||
    plan.profile !== 'coverage' ||
    !Number.isInteger(plan.action_batch) ||
    plan.action_batch < 1 ||
    plan.action_batch > 64 ||
    !Array.isArray(plan.trial_budgets) ||
    !plan.trial_budgets.length ||
    plan.trial_budgets.some(
      (value, i) =>
        !Number.isInteger(value) ||
        value < 1 ||
        value > 65536 ||
        (i && value <= plan.trial_budgets[i - 1]),
    ) ||
    !plan.trial_budgets.includes(budget) ||
    !Number.isFinite(plan.maximum_minutes) ||
    plan.maximum_minutes <= 0 ||
    plan.maximum_minutes > 50 ||
    typeof plan.stop_when_uncensored !== 'boolean' ||
    !Array.isArray(plan.rankings) ||
    !plan.rankings.length ||
    (plan.neural_origin !== undefined &&
      !['source', 'previous', 'both'].includes(plan.neural_origin)) ||
    (plan.preserve_uv !== undefined && typeof plan.preserve_uv !== 'boolean') ||
    new Set(plan.rankings).size !== plan.rankings.length ||
    plan.rankings.some(
      (value) => !['learned', 'constant', 'shuffled', 'shortest', 'current-plane'].includes(value),
    )
  )
    throw new Error('invalid bounded development comparison');
  return {
    ...visual,
    profile: plan.profile,
    action_trials: budget,
    action_batch: plan.action_batch,
    audit_rankings: plan.rankings,
    ...(plan.neural_origin === undefined ? {} : { neural_origin: plan.neural_origin }),
    ...(plan.preserve_uv === undefined ? {} : { preserve_uv: plan.preserve_uv }),
  };
}
export function compareBudgetRound(rows, rankings) {
  if (rows.length !== 2 || rows.some((row) => !successful(row)))
    throw new Error('incomplete matched budget');
  const comparisons = rankings.map((ranking) => ({
    ranking,
    ...compareAudits(rows[0].audit, rows[1].audit, ranking),
  }));
  const termination = rows.map((row) => auditTermination(row.audit));
  return {
    comparisons,
    termination,
    uncensored: termination.every((value) => value.known && value.uncensored),
  };
}
async function main() {
  const [planPath, initialModel, finalModel, directory] = process.argv.slice(2);
  if (!planPath || !initialModel || !finalModel || !directory || fs.existsSync(directory))
    throw new Error('budget-quality.mjs PLAN_JSON INITIAL_MODEL FINAL_MODEL NEW_OUTPUT');
  const plan = read(planPath),
    visual = read(plan.settings),
    manifest = read(plan.manifest);
  budgetSettings(plan, visual, plan.trial_budgets?.[0]);
  if (!manifest.assets?.length || manifest.assets.some((asset) => asset.split !== 'development'))
    throw new Error('budget calibration requires frozen development assets');
  const root = path.resolve(directory),
    models = [initialModel, finalModel].map((model) => ({
      path: path.resolve(model),
      sha256: digest(model),
    }));
  fs.mkdirSync(root, { recursive: true });
  write(root + '/manifest.json', manifest);
  write(root + '/plan.json', plan);
  const report = {
    version: 1,
    complete: false,
    score: null,
    release_quality_proven: false,
    models,
    plan_sha256: digest(planPath),
    manifest_sha256: digest(plan.manifest),
    rounds: [],
  };
  const started = Date.now(),
    deadline = started + plan.maximum_minutes * 60000,
    cancellation = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => cancellation.abort());
  try {
    for (const budget of plan.trial_budgets) {
      const round = { action_trials: budget, complete: false, rows: [] };
      report.rounds.push(round);
      const settings = root + '/budget-' + budget + '.json';
      write(settings, budgetSettings(plan, visual, budget));
      for (const [index, model] of models.entries()) {
        if (cancellation.signal.aborted || deadline <= Date.now())
          throw new Error('budget comparison deadline/cancellation');
        const output = root + '/budget-' + budget + '-model-' + index + '.json',
          fd = fs.openSync(output + '.log', 'w');
        try {
          const result = await boundedProcess(
            process.env.BLITZ_AUDIT_BINARY ?? 'build/neural/blitz-neural-diagnostics',
            ['--audit-model', root + '/manifest.json', model.path, settings, output],
            {
              stdio: ['ignore', fd, fd],
              maximum: deadline - Date.now(),
              signal: cancellation.signal,
            },
          );
          const audit = fs.existsSync(output) ? read(output) : null;
          round.rows.push({ model: index, ...result, audit });
          write(root + '/report.json', report);
          if (!successful(result))
            throw new Error('failed audit execution: ' + JSON.stringify(result));
          if (audit?.model_sha256 !== model.sha256 || digest(model.path) !== model.sha256)
            throw new Error('audited model changed or differs from requested checkpoint');
        } finally {
          fs.closeSync(fd);
        }
      }
      Object.assign(round, compareBudgetRound(round.rows, plan.rankings), { complete: true });
      write(root + '/report.json', report);
      if (plan.stop_when_uncensored && round.uncensored) {
        report.status = 'uncensored';
        break;
      }
    }
    report.complete = true;
    report.status ??= 'budget_ladder_complete';
  } catch (error) {
    report.error = String(error);
    report.status = cancellation.signal.aborted ? 'cancelled' : 'incomplete';
    process.exitCode = 1;
  } finally {
    report.seconds = (Date.now() - started) / 1000;
    write(root + '/report.json', report);
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
