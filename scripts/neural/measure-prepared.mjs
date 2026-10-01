// Paired local diagnostic, not a training campaign or quality SCORE.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { hash, modelPayload } from './refactor-cycle.mjs';
const [root, model, checkpoint, control] = process.argv.slice(2);
if (!root || !model || !checkpoint || fs.existsSync(root))
  throw Error('measure-prepared.mjs FRESH_DIRECTORY MODEL CHECKPOINT');
fs.mkdirSync(root, { recursive: true });
const variants =
  control === 'native'
    ? [
        { name: 'batch2', batch: 2, backend: 'reference' },
        { name: 'combined', batch: 2, backend: 'fused' },
      ]
    : control
      ? [
          { name: 'control', batch: 1, backend: 'reference', binary: control },
          { name: 'serial', batch: 1, backend: 'reference' },
          { name: 'batch2', batch: 2, backend: 'reference' },
          { name: 'combined', batch: 2, backend: 'fused' },
        ]
      : [
          { name: 'serial', batch: 1, backend: 'reference' },
          { name: 'batch2', batch: 2, backend: 'reference' },
          { name: 'batch4', batch: 4, backend: 'reference' },
          { name: 'combined', batch: 4, backend: 'fused' },
        ];
const result = {
  complete: false,
  score: null,
  storage: 'fp32 control (packed quality failures measured separately)',
  binary_sha256: hash('build/neural/blitz-neural-cycle'),
  model_sha256: hash(model),
  checkpoint_sha256: hash(checkpoint),
  runs: [],
};
for (let repeat = 0; repeat < 5; repeat++)
  for (const v of repeat % 2 ? [...variants].reverse() : variants) {
    const dir = root + '/' + v.name + '-' + repeat,
      fd = fs.openSync(dir + '.log', 'w');
    const start = performance.now();
    const args = [
      dir,
      '--curriculum',
      'research/neural/prepared-pilot/timing-curriculum.json',
      '--states',
      '8',
      '--updates',
      '2048',
      '--minutes',
      '2',
      '--gpu-memory-mib',
      '1024',
      '--vertex-storage',
      'fp32',
      '--view-batch',
      '1',
      '--candidate-batch',
      String(v.batch),
      '--update-backend',
      v.backend,
      '--initialize',
      model,
      '--warmstart',
      checkpoint,
    ];
    const proc = spawnSync(v.binary ?? 'build/neural/blitz-neural-cycle', args, {
      stdio: ['ignore', fd, fd],
      timeout: 130000,
    });
    fs.closeSync(fd);
    const report = fs.existsSync(dir + '/report.json')
      ? JSON.parse(fs.readFileSync(dir + '/report.json'))
      : null;
    const row = {
      variant: v.name,
      binary_sha256: hash(v.binary ?? 'build/neural/blitz-neural-cycle'),
      repeat,
      exit: proc.status,
      process_seconds: (performance.now() - start) / 1000,
      report,
    };
    if (report?.complete) {
      row.payload = modelPayload(dir + '/' + report.checkpoint + '/model.blzn');
      row.shards = report.datasets.map((x) => hash(dir + '/data/' + x + '/actions.bin'));
    }
    result.runs.push(row);
    fs.writeFileSync(root + '/report.json', JSON.stringify(result, null, 2) + '\n');
    console.log(
      JSON.stringify({
        variant: v.name,
        repeat,
        complete: report?.complete,
        seconds: report?.seconds,
      }),
    );
    if (!report?.complete) throw Error('Diagnostic incomplete; inspect retained evidence');
  }
for (let repeat = 0; repeat < 5; repeat++) {
  const rows = result.runs.filter((x) => x.repeat === repeat && x.variant !== 'combined');
  if (
    new Set(rows.map((x) => x.payload)).size !== 1 ||
    new Set(rows.map((x) => JSON.stringify(x.shards))).size !== 1
  )
    throw Error('Candidate batching changed teacher labels or model payload');
}
result.serial_batch_exact = true;
result.complete = true;
fs.writeFileSync(root + '/report.json', JSON.stringify(result, null, 2) + '\n');
