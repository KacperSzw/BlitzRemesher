// Width selection consumes complete, matched learning and mesh-quality evidence.
import fs from 'node:fs';
import path from 'node:path';
import { boundedProcess } from './bounded-process.mjs';
import { fileURLToPath } from 'node:url';
import { read, write } from './artifacts.mjs';
import { compareAudits, successful } from './quality-metrics.mjs';
export function widthDecision(rows, seeds) {
  if (seeds.length !== 3 || new Set(seeds).size !== 3)
    throw new Error('three independent seeds required');
  const result = { selected_width: 64, passed: false, score: null, comparisons: [] };
  for (const width of [128, 256]) {
    const comparison = { width, passed: true, seeds: [] };
    for (const seed of seeds) {
      const bases = rows.filter((row) => row.seed === seed && row.width === 64),
        candidates = rows.filter((row) => row.seed === seed && row.width === width);
      try {
        if (
          bases.length !== 1 ||
          candidates.length !== 1 ||
          !successful(bases[0]) ||
          !successful(candidates[0])
        )
          throw new Error('missing or failed execution');
        const metrics = compareAudits(bases[0].audit, candidates[0].audit);
        // Quality is primary. Time/memory are diagnostics, not gates or rewards.
        const passed = metrics.retained_ratio_improvement > 0;
        comparison.seeds.push({ seed, complete: true, passed, ...metrics });
        comparison.passed &&= passed;
      } catch (error) {
        comparison.passed = false;
        comparison.seeds.push({ seed, complete: false, error: String(error) });
      }
    }
    result.comparisons.push(comparison);
    if (comparison.passed && !result.passed) {
      result.selected_width = width;
      result.passed = true;
    }
  }
  return result;
}
async function main() {
  const [input, directory, updatesText = '128', settingsPath] = process.argv.slice(2),
    root = path.resolve(directory ?? ''),
    updates = Number(updatesText);
  if (
    !input ||
    !directory ||
    !settingsPath ||
    ![32, 128, 512, 2048].includes(updates) ||
    fs.existsSync(root)
  )
    throw new Error(
      'width-quality.mjs ABLATION_DIRECTORY NEW_OUTPUT UPDATES FROZEN_AUDIT_SETTINGS',
    );
  fs.mkdirSync(root, { recursive: true });
  const started = Date.now(),
    deadline = Number(process.env.BLITZ_VALIDATION_DEADLINE ?? started + 3000000),
    seeds = [101, 211, 307];
  const report = {
      complete: false,
      score: null,
      updates,
      rows: [],
      gate: widthDecision([], seeds),
    },
    cancellation = new AbortController();
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => cancellation.abort());
  const settings = read(settingsPath);
  write(root + '/settings.json', settings);
  try {
    for (const seed of seeds)
      for (const width of [64, 128, 256]) {
        if (cancellation.signal.aborted || Date.now() + 5000 >= deadline)
          throw new Error('quality comparison deadline');
        const label = `seed${seed}-w${width}-u${updates}`,
          cycle = read(path.resolve(input, label, 'report.json'));
        if (!cycle.complete) throw new Error('incomplete learning ablation ' + label);
        const output = root + '/' + label + '.json',
          fd = fs.openSync(root + '/' + label + '.log', 'w');
        try {
          const result = await boundedProcess(
            'build/neural/blitz-neural-diagnostics',
            [
              '--audit-model',
              'research/neural/corpus-v2/validation.json',
              path.resolve(input, label, cycle.checkpoint, 'model.blzn'),
              root + '/settings.json',
              output,
            ],
            {
              stdio: ['ignore', fd, fd],
              maximum: deadline - Date.now() - 3000,
              signal: cancellation.signal,
            },
          );
          report.rows.push({
            seed,
            width,
            ...result,
            audit: fs.existsSync(output) ? read(output) : null,
          });
          write(root + '/report.json', report);
          if (!result.success)
            throw new Error('quality audit failed ' + label + ': ' + JSON.stringify(result));
        } finally {
          fs.closeSync(fd);
        }
      }
    report.complete =
      report.rows.length === 9 &&
      report.rows.every((row) => successful(row) && row.audit?.complete === true);
  } catch (error) {
    report.error = String(error);
    process.exitCode = 1;
  } finally {
    report.gate = widthDecision(report.rows, seeds);
    report.seconds = (Date.now() - started) / 1000;
    write(root + '/report.json', report);
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
