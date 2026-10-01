#!/usr/bin/env node
// A bounded local measurement of one pass through the existing curriculum.
// Keeps partial runs visible; never assigns a quality score or promotes a model.
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import {
  cycleConditions,
  datasetHealth,
  trainingHealth,
  modelPayload,
  hash,
} from './refactor-cycle.mjs';
import { auditRowsHealthy } from './action-gates.mjs';
const [directory, initialModel, initialCheckpoint, startStepText] = process.argv.slice(2);
const initialStep = Number(startStepText);
if (
  !directory ||
  !initialModel ||
  !initialCheckpoint ||
  !Number.isSafeInteger(initialStep) ||
  initialStep < 0
)
  throw new Error('pipeline-profile.mjs FRESH_DIRECTORY MODEL CHECKPOINT START_STEP');
if (fs.existsSync(directory)) throw new Error('Choose a fresh output directory');
fs.mkdirSync(directory + '/data', { recursive: true });
const root = path.resolve(directory),
  read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const report = {
  revision: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  score: null,
  complete: false,
  precision: 'IEEE FP32; existing FP64 audit arithmetic retained',
  started: new Date().toISOString(),
  initial_model_sha256: hash(initialModel),
  initial_checkpoint_sha256: hash(initialCheckpoint),
  initial_step: initialStep,
  states_per_condition: 8,
  pool: 4,
  updates_per_segment: 2048,
  batch: 512,
  teacher_memory_mib: 1024,
  conditions: cycleConditions,
  phases: [],
};
const save = () => fs.writeFileSync(root + '/report.json', JSON.stringify(report, null, 2) + '\n');
const gpu = () =>
  execFileSync(
    'nvidia-smi',
    [
      '--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu',
      '--format=csv,noheader',
    ],
    { encoding: 'utf8' },
  ).trim();
async function execute(phase, binary, args, name) {
  const fd = fs.openSync(root + '/' + name + '.log', 'w'),
    before = gpu(),
    start = performance.now();
  const row = { phase, binary, binary_sha256: hash(binary), args, gpu_before: before };
  report.phases.push(row);
  save();
  const child = spawn(binary, args, { stdio: ['ignore', fd, fd] });
  let hard;
  const timeout = setTimeout(() => {
    row.timed_out = true;
    child.kill('SIGTERM');
    hard = setTimeout(() => child.kill('SIGKILL'), 5000);
  }, 150000);
  try {
    row.code = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', resolve);
    });
  } finally {
    clearTimeout(timeout);
    clearTimeout(hard);
    fs.closeSync(fd);
    row.wall_seconds = (performance.now() - start) / 1000;
    row.gpu_after = gpu();
    save();
  }
  console.log(JSON.stringify({ phase, name, code: row.code, wall_seconds: row.wall_seconds }));
  return row;
}
const began = performance.now();
save();
let model = path.resolve(initialModel),
  checkpoint = path.resolve(initialCheckpoint),
  step = initialStep,
  datasets = [];
try {
  for (const [i, c] of cycleConditions.entries()) {
    const name = 'shard-' + i,
      out = root + '/data/' + name;
    const teacher = await execute(
      'teacher',
      'build/neural/blitz-neural-placement-prepare',
      [
        c.asset,
        out,
        '--states',
        '8',
        '--pool',
        '4',
        '--pixels',
        String(c.pixels),
        '--previous-steps',
        String(c.previous),
        '--seed',
        String(101 + i),
        '--model',
        model,
        '--gpu-memory-mib',
        '1024',
        '--minutes',
        '2',
      ],
      name,
    );
    teacher.condition = c;
    if (fs.existsSync(out + '/index.json')) teacher.index = read(out + '/index.json');
    if (fs.existsSync(out + '/reuse.json')) teacher.reuse = read(out + '/reuse.json');
    teacher.healthy = teacher.code === 0 && datasetHealth(out, !!c.previous);
    save();
    if (!teacher.healthy) throw new Error('Incomplete teacher condition ' + i);
    datasets.push(name);
    fs.writeFileSync(root + '/data/index.json', JSON.stringify({ datasets }, null, 2) + '\n');
    const train = root + '/train-' + i,
      target = step + 2048;
    const update = await execute(
      'training',
      'build/neural/blitz-neural-action-train',
      [
        root + '/data',
        train,
        '--warmstart',
        checkpoint,
        '--steps',
        String(target),
        '--batch',
        '512',
        '--seed',
        '101',
        '--minutes',
        '1',
      ],
      'train-' + i,
    );
    if (update.code !== 0) throw new Error('Incomplete training segment ' + i);
    update.health = trainingHealth(train, target);
    update.model_payload_sha256 = modelPayload(train + '/' + update.health.model);
    step = target;
    checkpoint = train + '/' + update.health.checkpoint;
    model = train + '/' + update.health.model;
    save();
  }
  const manifest = 'research/neural/action-diagnostic.json',
    config = 'research/neural/refactor-smoke.json',
    assets = read(manifest).assets;
  for (const ranking of ['constant', 'learned']) {
    const out = root + '/audit-' + ranking;
    const row = await execute(
      'quality_audit',
      'build/neural/blitz',
      [
        'bench',
        manifest,
        config,
        out,
        '--neural-model',
        model,
        '--neural-control',
        ranking,
        '--ranking-seed',
        '101',
        '--action-trials',
        '8',
        '--action-batch',
        '16',
        '--gpu-memory-mib',
        '1024',
        '--neural-confirmation',
        'gpu',
        '--minutes',
        '2',
      ],
      'audit-' + ranking,
    );
    if (fs.existsSync(out + '/summary.json')) row.summary = read(out + '/summary.json');
    row.assets = assets
      .map((a) => out + '/rows/' + a.id + '.json')
      .filter(fs.existsSync)
      .map(read);
    row.healthy = row.code === 0 && row.summary?.complete && auditRowsHealthy(row.assets, assets);
    save();
    if (!row.healthy) throw new Error('Incomplete quality diagnostic ' + ranking);
  }
  report.complete = true;
  report.final_step = step;
  report.states = datasets.reduce(
    (n, p) => n + read(root + '/data/' + p + '/index.json').states,
    0,
  );
  report.final_model = model;
  report.final_checkpoint = checkpoint;
} catch (error) {
  report.error = String(error);
  process.exitCode = 1;
} finally {
  report.wall_seconds = (performance.now() - began) / 1000;
  report.finished = new Date().toISOString();
  report.child_seconds = report.phases.reduce((n, p) => n + (p.wall_seconds ?? 0), 0);
  save();
}
