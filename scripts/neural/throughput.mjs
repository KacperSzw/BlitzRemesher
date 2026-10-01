// Bounded, fixed-work comparisons. A frozen teacher keeps labels independent of
// worker scheduling; adaptive learning experiments use a separate mode.
import fs from 'node:fs';
import path from 'node:path';
import { boundedProcess } from './bounded-process.mjs';
import { read, write } from './artifacts.mjs';
const [mode, directory] = process.argv.slice(2),
  root = path.resolve(directory ?? '');
if (
  !['smoke', 'matrix', 'pilot', 'expanded', 'ablation', 'seeds'].includes(mode) ||
  !directory ||
  fs.existsSync(root)
)
  throw new Error('throughput.mjs smoke|matrix|pilot|expanded|ablation|seeds NEW_OUTPUT');
fs.mkdirSync(root, { recursive: true });
const start = Date.now(),
  deadline = Number(process.env.BLITZ_VALIDATION_DEADLINE ?? start + 600000),
  memory = Number(process.env.BLITZ_VALIDATION_MEMORY_MIB ?? 512);
const binary = process.env.BLITZ_CYCLE_BINARY ?? 'build/neural/blitz-neural-cycle',
  report = { complete: false, score: null, mode, started: start, rows: [] },
  cancellation = new AbortController();
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => cancellation.abort());
const assets = ['ph_painted_wooden_bench', 'ph_sweet_potato'];
const conditions = Array.from({ length: 6 }, (_, i) => ({
  asset: assets[i % 2],
  pixels: i % 2 ? 64 : 32,
  previous_steps: 0,
}));
write(root + '/curriculum.json', { version: 1, conditions });
async function cycle(label, options, { allowFailure = false } = {}) {
  if (cancellation.signal.aborted || Date.now() + 5000 >= deadline)
    throw new Error('comparison deadline reached');
  const out = root + '/' + label;
  const args = [
    out,
    '--minutes',
    '3',
    '--states',
    process.env.BLITZ_COMPARISON_STATES ?? '2',
    '--updates',
    '128',
    '--quality',
    'off',
    '--training-profile',
    'coverage',
    '--update-backend',
    'fused',
    '--curriculum',
    root + '/curriculum.json',
    '--gpu-memory-mib',
    String(memory),
    '--frozen-teacher',
    'on',
    ...options,
  ];
  const fd = fs.openSync(out + '.log', 'w'),
    begin = Date.now();
  try {
    const result = await boundedProcess(binary, args, {
      stdio: ['ignore', fd, fd],
      maximum: Math.min(
        Number(args[args.lastIndexOf('--minutes') + 1]) * 60000 + 10000,
        deadline - Date.now() - 5000,
      ),
      signal: cancellation.signal,
    });
    const row = {
      label,
      args,
      ...result,
      wall_seconds: (Date.now() - begin) / 1000,
      report: fs.existsSync(out + '/report.json') ? read(out + '/report.json') : null,
    };
    report.rows.push(row);
    write(root + '/report.json', report);
    if (!result.success && !allowFailure)
      throw new Error(label + ' failed: ' + JSON.stringify(result));
    return row;
  } finally {
    fs.closeSync(fd);
  }
}
try {
  if (mode === 'pilot') {
    await cycle(
      'pilot',
      [
        '--curriculum',
        'research/neural/prepared-pilot/curriculum.json',
        '--states',
        '1',
        '--updates',
        '32',
        '--workers',
        '2',
        '--candidate-batch',
        '4',
        '--minutes',
        '5',
      ],
      { allowFailure: true },
    );
  } else if (mode === 'expanded') {
    await cycle(
      'qualification',
      [
        '--curriculum',
        'research/neural/corpus-v2/curriculum.json',
        '--corpus',
        'research/neural/corpus-v2/corpus.json',
        '--training-selection',
        'research/neural/corpus-v2/training.json',
        '--states',
        '1',
        '--updates',
        '32',
        '--workers',
        '3',
        '--candidate-batch',
        '4',
        '--minutes',
        '15',
      ],
      { allowFailure: true },
    );
  } else if (mode === 'seeds') {
    for (const seeds of ['on', 'off'])
      await cycle('simplifier-' + seeds, [
        '--states',
        '2',
        '--passes',
        '4',
        '--updates',
        '128',
        '--workers',
        '2',
        '--candidate-batch',
        '4',
        '--episode-seeds',
        'on',
        '--simplifier-seeds',
        seeds,
        '--frozen-teacher',
        'off',
      ]);
  } else if (mode === 'ablation') {
    for (const seed of [101, 211, 307])
      for (const width of [64, 128, 256])
        for (const updates of [32, 128, 512, 2048])
          await cycle(`seed${seed}-w${width}-u${updates}`, [
            '--seed',
            String(seed),
            '--hidden-width',
            String(width),
            '--updates',
            String(updates),
            '--workers',
            '2',
            '--candidate-batch',
            '4',
            '--frozen-teacher',
            'off',
          ]);
  } else {
    for (const workers of mode === 'smoke' ? [1, 2] : [1, 2, 3])
      for (const batch of mode === 'smoke' ? [2] : [1, 2, 4, 8])
        await cycle(`workers${workers}-batch${batch}`, [
          '--workers',
          String(workers),
          '--candidate-batch',
          String(batch),
        ]);
    const first = report.rows[0]?.report?.datasets;
    if (!first?.length) throw new Error('no complete comparison pages');
    const hashes = (row) =>
      row.report.datasets.map(
        (name) => read(root + '/' + row.label + '/data/' + name + '/index.json').sha256,
      );
    const reference = hashes(report.rows[0]);
    for (const row of report.rows)
      if (JSON.stringify(reference) !== JSON.stringify(hashes(row)))
        throw new Error('fixed-work labels changed across worker/batch configurations');
    report.identical_labels = true;
  }
  report.complete = report.rows.every((row) => row.success && row.report?.complete === true);
} catch (error) {
  report.error = String(error);
  process.exitCode = 1;
} finally {
  report.seconds = (Date.now() - start) / 1000;
  write(root + '/report.json', report);
  console.log(
    JSON.stringify({ complete: report.complete, seconds: report.seconds, error: report.error }),
  );
}
