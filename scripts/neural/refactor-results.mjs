#!/usr/bin/env node
// Local, credential-free collection/reporting. --watch survives a chat ending
// when launched as a user service; it never creates or modifies cloud resources.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { read, write } from './artifacts.mjs';
const [directory, mode] = process.argv.slice(2),
  dir = path.resolve(directory ?? '');
if (!directory || (mode && mode !== '--watch'))
  throw new Error('refactor-results.mjs RUN_DIRECTORY [--watch]');
const report = dir + '/RESULTS.md';
async function hash(file) {
  const h = crypto.createHash('sha256');
  for await (const b of fs.createReadStream(file)) h.update(b);
  return h.digest('hex');
}
const sleep = () => new Promise((resolve) => setTimeout(resolve, 30000));
function pending(s) {
  fs.writeFileSync(
    report,
    `# GPU refactor run\n\nStatus: ${s.phase ?? 'starting'}.\n\nThe controller owns validation, the two-hour learning cycle, checksum-verified collection and GPU termination. This report is replaced after collection. The live architecture and measurements are in [ARCHITECTURE.md](live/ARCHITECTURE.md).\n\nHard rental deadline: ${new Date(s.deadline_ms).toISOString()}. Additional authorization: $${s.budget.authorization.additional_usd}.\n`,
  );
}
try {
  for (;;) {
    const s = read(dir + '/rental.json');
    if (
      fs.existsSync(dir + '/collection.json') &&
      s.compute_terminated &&
      (s.volume_deleted || s.phase === 'stopped_uncollected')
    ) {
      const collection = read(dir + '/collection.json'),
        archive = dir + '/results.tar.gz';
      if (!collection.verified || (await hash(archive)) !== collection.sha256)
        throw new Error('Collected archive checksum mismatch');
      const entries = execFileSync('tar', ['-tzf', archive], {
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
      })
        .trim()
        .split('\n');
      if (entries.some((p) => !p.startsWith('results/') || p.split('/').includes('..')))
        throw new Error('Unexpected result archive path');
      const output = dir + '/evidence';
      fs.mkdirSync(output, { recursive: true });
      execFileSync('tar', [
        '--no-same-owner',
        '--no-same-permissions',
        '-xzf',
        archive,
        '-C',
        output,
      ]);
      const root = output + '/results/gpu-refactor',
        result = read(root + '/result.json'),
        cycle = fs.existsSync(root + '/cycle.json') ? read(root + '/cycle.json') : null;
      const summary = {
        revision: s.revision,
        archive_sha256: collection.sha256,
        validation_passed: result.validation_passed,
        complete: result.complete,
        quality_proven: false,
        compute_terminated: s.compute_terminated,
        volume_deleted: !!s.volume_deleted,
        conservative_rental_usd:
          ((s.terminated_at - s.started_at) / 3600000) * (s.deployment.gpu_hourly_usd_cap + 0.01),
        error: result.error ?? null,
      };
      let details = '';
      if (cycle) {
        const training = cycle.stages.filter((s) => s.phase === 'training'),
          preparation = cycle.stages.filter((s) => s.phase === 'preparation'),
          last = training.at(-1)?.health;
        Object.assign(summary, {
          learning_seconds: cycle.active_seconds ?? null,
          optimizer_updates: cycle.training_steps,
          training_states: last?.states ?? null,
          accepted_shards: cycle.datasets.length,
          excluded_shards: preparation.filter((s) => !s.usable).length,
          matched_audits: cycle.audits.length,
        });
        if (result.complete && cycle.complete && last) {
          const local = (remote) => {
            if (
              !remote.startsWith('/workspace/results/gpu-refactor/') ||
              remote.split('/').includes('..')
            )
              throw new Error('Unexpected checkpoint path');
            return output + remote.slice('/workspace'.length);
          };
          const model = local(cycle.model),
            checkpoint = local(cycle.checkpoint);
          if (
            (await hash(model)) !== last.model_sha256 ||
            (await hash(checkpoint)) !== last.checkpoint_sha256
          )
            throw new Error('Final model/checkpoint checksum mismatch');
          fs.copyFileSync(model, dir + '/final-model.blzn');
          fs.copyFileSync(checkpoint, dir + '/final-checkpoint.pt');
          summary.model_sha256 = last.model_sha256;
          details +=
            '\nFinal artifacts: [model](final-model.blzn), [resumable optimizer checkpoint](final-checkpoint.pt).\n';
        }
        const final = cycle.audits.find((a) => a.final)?.report;
        if (final) {
          details +=
            '\nFinal fixed development diagnostic (ratio 1 means unreduced):\n\n| Asset | Constant ratio | Learned ratio |\n| --- | ---: | ---: |\n';
          for (const row of final.runs.find((r) => r.ranking === 'learned').rows) {
            const other = final.runs
              .find((r) => r.ranking === 'constant')
              .rows.find((r) => r.id === row.id);
            details += `| ${row.id} | ${other.ratio.toFixed(6)} | ${row.ratio.toFixed(6)} |\n`;
          }
        }
      }
      write(dir + '/summary.json', summary);
      fs.writeFileSync(
        report,
        `# GPU refactor run\n\n${summary.complete ? 'The learning cycle completed.' : 'The run is incomplete; inspect the retained error and evidence.'} Remote validation: ${summary.validation_passed ? 'passed' : 'failed'}. GPU terminated: ${summary.compute_terminated}. Storage deleted after verified collection: ${summary.volume_deleted}.\n\nUpdates: ${summary.optimizer_updates ?? 0}; training states: ${summary.training_states ?? 0}; accepted/excluded shards: ${summary.accepted_shards ?? 0}/${summary.excluded_shards ?? 0}; matched audits: ${summary.matched_audits ?? 0}.\n\nConservative rental ledger: $${summary.conservative_rental_usd.toFixed(3)} against the $8 additional grant. This uses the capped hourly rate, not a final provider invoice.\n${details}\n[Architecture and timings](evidence/results/gpu-refactor/ARCHITECTURE.md) · [Machine-readable summary](summary.json) · [Full result](evidence/results/gpu-refactor/result.json)\n\nThis is a two-asset screening diagnostic. It does not establish a full-protocol score, held-out generalization or global optimality.\n${summary.error ? '\nError: ' + summary.error + '\n' : ''}`,
      );
      break;
    }
    pending(s);
    if (s.compute_terminated && !fs.existsSync(dir + '/collection.json')) {
      fs.appendFileSync(
        report,
        '\nCompute stopped before verified collection. The controller preserves uncollected recovery storage.\n',
      );
      break;
    }
    if (mode !== '--watch') break;
    await sleep();
  }
} catch (error) {
  fs.appendFileSync(report, '\nReporting error: ' + String(error) + '\n');
  throw error;
}
