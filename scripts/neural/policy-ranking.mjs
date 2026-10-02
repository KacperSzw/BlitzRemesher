// Local policy-alignment experiment. This driver has no provider/rental actions.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { boundedProcess } from './bounded-process.mjs';
import { comparePolicyRanking } from './policy-ranking-quality.mjs';
import { rankingSupport } from './policy-ranking-support.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (p) => JSON.parse(fs.readFileSync(p));
const hash = (p) => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const write = (p, value) => {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(value, null, 2) + '\n');
};

export async function runPolicyRanking(configPath, directory) {
  process.chdir(root);
  const config = read(configPath);
  if (fs.existsSync(directory)) throw Error('Choose a fresh policy-ranking experiment directory');
  if (config.version !== 1 || !config.training_assets.length || !config.seeds.length)
    throw Error('Invalid policy-ranking experiment configuration');
  const objective = config.objective ?? 'policy-ranking';
  if (!['policy-ranking', 'endpoint-ranking'].includes(objective))
    throw Error('Invalid ranking objective');
  if (objective === 'endpoint-ranking' && config.rank_fraction !== undefined)
    throw Error('Endpoint scorer cannot use ranking-row blending');
  if (
    config.rank_fraction !== undefined &&
    (!Number.isFinite(config.rank_fraction) || config.rank_fraction < 0 || config.rank_fraction > 1)
  )
    throw Error('Ranking adjustment fraction must be within [0,1]');
  const teacher = config.runtime_teacher
    ? 'build/neural/blitz-neural-policy-prepare'
    : 'build/neural/blitz-neural-placement-prepare';
  const trainer = 'build/neural/blitz-neural-action-train';
  const diagnostic = 'build/neural/blitz-neural-diagnostics';
  const deadline = Date.now() + (config.maximum_minutes ?? 45) * 60000;
  const training = read(config.training_selection).assets;
  const auditAssets = read(config.audit_manifest).assets;
  const reservedAssets = (config.reserved_evaluation_manifests ?? []).flatMap(
    (p) => read(p).assets,
  );
  for (const id of config.training_assets) {
    const asset = training.find((a) => a.id === id && a.split === 'development');
    if (!asset || !asset.source_group)
      throw Error('Training asset outside development selection: ' + id);
    if (
      [...auditAssets, ...reservedAssets].some(
        (a) => a.id === id || a.source_group === asset.source_group,
      )
    )
      throw Error('Training/evaluation asset overlap: ' + id);
  }
  const identity = {
    config,
    config_sha256: hash(configPath),
    model_sha256: hash(config.initial_model),
    revision: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    dirty_patch_sha256: createHash('sha256')
      .update(execFileSync('git', ['diff', 'HEAD']))
      .digest('hex'),
    binaries: Object.fromEntries([teacher, trainer, diagnostic].map((p) => [p, hash(p)])),
    manifests: Object.fromEntries(
      [
        config.corpus,
        config.training_selection,
        config.audit_manifest,
        config.audit_settings,
        ...(config.reserved_evaluation_manifests ?? []),
        'research/PROTOCOL.md',
      ].map((p) => [p, hash(p)]),
    ),
    started_at: new Date().toISOString(),
    timing_authority: 'local_shared',
  };
  const sourcePaths = execFileSync(
    'git',
    [
      'ls-files',
      '--cached',
      '--others',
      '--exclude-standard',
      '--',
      'src',
      'training',
      'tools/neural',
      'include',
      'cmake',
      'scripts/neural',
      'CMakeLists.txt',
    ],
    { encoding: 'utf8' },
  )
    .trim()
    .split('\n');
  identity.source_files = Object.fromEntries(
    [...new Set(sourcePaths)].sort().map((p) => [p, hash(p)]),
  );
  write(path.join(directory, 'identity.json'), identity);
  const report = {
    complete: false,
    remote_ready: false,
    score: null,
    tasks: [],
    data: [],
    training: [],
    audits: [],
  };
  const persist = () => write(path.join(directory, 'report.json'), report);
  const env = {
    ...process.env,
    VK_DRIVER_FILES: '/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json',
  };
  for (const key of ['DISPLAY', 'WAYLAND_DISPLAY', 'VK_ICD_FILENAMES']) delete env[key];
  async function run(name, command, args, maximum) {
    if (hash(command) !== identity.binaries[command]) throw Error('Experiment binary changed');
    const log = path.join(directory, 'logs', name + '.log');
    fs.mkdirSync(path.dirname(log), { recursive: true });
    const fd = fs.openSync(log, 'wx'),
      started = Date.now();
    let process;
    try {
      process = await boundedProcess(command, args, {
        maximum: Math.min(maximum, deadline - started),
        env,
        stdio: ['ignore', fd, fd],
      });
    } finally {
      fs.closeSync(fd);
    }
    const task = {
      name,
      command,
      args,
      maximum,
      process,
      seconds: (Date.now() - started) / 1000,
      log_sha256: hash(log),
    };
    report.tasks.push(task);
    persist();
    console.log(JSON.stringify({ task: name, success: process.success, seconds: task.seconds }));
    if (!process.success) throw Error('Experiment task failed: ' + name);
  }
  try {
    const dataRoot = config.dataset_directory ?? path.join(directory, 'data'),
      datasets = [];
    if (config.runtime_teacher && !config.dataset_directory)
      for (const asset of config.training_assets)
        for (const preserve of [true, false])
          for (const ranking of config.runtime_teacher.rollout_rankings ?? ['learned'])
            for (const sourceTriangles of config.runtime_teacher.source_triangles ?? [0]) {
              const name = `${asset}-uv${Number(preserve)}-${ranking}-source${sourceTriangles}`;
              const jobPath = path.join(directory, 'jobs', name + '.json');
              const output = path.join(dataRoot, name);
              write(jobPath, {
                asset,
                model: config.initial_model,
                corpus: config.corpus,
                training_selection: config.training_selection,
                settings: config.audit_settings,
                preserve_uv: preserve,
                gpu_memory_mib: config.gpu_memory_mib,
                rollout_ranking: ranking,
                source_triangles: sourceTriangles,
                source_output: config.runtime_teacher.source_output ?? 'rebuild',
                source_merge_wedges: config.runtime_teacher.source_merge_wedges ?? false,
                teacher_cache_rasters: config.runtime_teacher.cache_rasters ?? false,
                states_per_proposal: config.runtime_teacher.states_per_proposal,
                maximum_states: config.runtime_teacher.maximum_states,
                pool: 16,
                seed: config.teacher_seed,
                minutes: config.teacher_minutes ?? 3,
              });
              await run(
                name,
                teacher,
                [jobPath, output],
                ((config.teacher_minutes ?? 3) * 60 + 30) * 1000,
              );
              const index = read(path.join(output, 'index.json'));
              if (
                !index.complete ||
                index.sha256 !== hash(path.join(output, 'actions.bin')) ||
                index.contract_sha256 !== hash(path.join(output, 'contract.json')) ||
                index.geometry_rejected_sha256 !==
                  hash(path.join(output, 'geometry-rejected.bin')) ||
                index.requests_sha256 !== hash(path.join(output, 'requests.json')) ||
                index.trajectory_sha256 !== hash(path.join(output, 'trajectory.json')) ||
                !index.verification.same_meshes_and_errors ||
                !index.verification.same_action_counts
              )
                throw Error('Unverified runtime teacher: ' + name);
              datasets.push(name);
              report.data.push({
                name,
                index_sha256: hash(path.join(output, 'index.json')),
                index,
              });
              persist();
            }
    for (const asset of config.dataset_directory || config.runtime_teacher
      ? []
      : config.training_assets)
      for (const condition of config.conditions ??
        config.pixels.map((pixels) => ({ pixels, source_limit: 3, adjacent_limit: 2 })))
        for (const preserve of [true, false])
          for (const retained of config.retained) {
            const { pixels, source_limit, adjacent_limit } = condition;
            const name = `${asset}-${pixels}-uv${Number(preserve)}-r${retained}`;
            const output = path.join(dataRoot, name);
            await run(
              name,
              teacher,
              [
                asset,
                output,
                '--architecture',
                '4',
                '--teacher-target',
                'policy-placement',
                '--teacher-selection',
                config.teacher_selection ?? 'geometric-random',
                '--training-profile',
                'coverage',
                '--mask-only-coverage',
                'on',
                '--states',
                String(config.states),
                '--pool',
                '16',
                '--pixels',
                String(pixels),
                '--source-limit',
                String(source_limit),
                '--adjacent-limit',
                String(adjacent_limit),
                '--audit-settings',
                config.audit_settings,
                '--mesh-cache',
                path.join(directory, 'mesh-cache'),
                '--preserve-uv',
                preserve ? 'on' : 'off',
                '--model',
                config.initial_model,
                '--raster-backend',
                'vulkan',
                '--vertex-storage',
                'packed',
                '--gpu-memory-mib',
                String(config.gpu_memory_mib),
                '--minutes',
                '3',
                '--seed',
                String(config.teacher_seed),
                '--corpus',
                config.corpus,
                '--training-selection',
                config.training_selection,
                '--simplifier-seed',
                retained < 1 ? 'on' : 'off',
                '--retained',
                String(retained),
              ],
              210000,
            );
            const index = read(path.join(output, 'index.json'));
            if (
              !index.complete ||
              index.sha256 !== hash(path.join(output, 'actions.bin')) ||
              index.geometry_rejected_sha256 !== hash(path.join(output, 'geometry-rejected.bin'))
            )
              throw Error('Unverified policy teacher data: ' + name);
            datasets.push(name);
            report.data.push({ name, index_sha256: hash(path.join(output, 'index.json')), index });
            persist();
          }
    if (config.dataset_directory) {
      for (const name of read(path.join(dataRoot, 'index.json')).datasets) {
        if (path.isAbsolute(name) || name.split(path.sep).includes('..'))
          throw Error('Invalid dataset path');
        const output = path.join(dataRoot, name),
          index = read(path.join(output, 'index.json'));
        if (
          !config.training_assets.includes(index.asset) ||
          auditAssets.some((a) => a.id === index.asset) ||
          !index.complete ||
          index.sha256 !== hash(path.join(output, 'actions.bin')) ||
          index.contract_sha256 !== hash(path.join(output, 'contract.json')) ||
          index.geometry_rejected_sha256 !== hash(path.join(output, 'geometry-rejected.bin'))
        )
          throw Error('Unverified reused policy teacher data: ' + name);
        if (
          objective === 'endpoint-ranking' &&
          (read(path.join(output, 'contract.json')).teacher_target !== 'runtime-endpoint-v3' ||
            index.verification?.same_meshes_and_errors !== true ||
            index.verification?.same_action_counts !== true ||
            index.verification?.same_trajectory_iterations !== true ||
            index.verification?.unknown_observation !== false ||
            index.requests_sha256 !== hash(path.join(output, 'requests.json')) ||
            index.trajectory_sha256 !== hash(path.join(output, 'trajectory.json')) ||
            index.source_augmentation?.reference_sha256 !==
              hash(path.join(output, 'source-reference.bin')))
        )
          throw Error('Unverified reused endpoint trajectory: ' + name);
        report.data.push({ name, index_sha256: hash(path.join(output, 'index.json')), index });
      }
      persist();
    } else write(path.join(dataRoot, 'index.json'), { datasets });
    if (config.runtime_teacher) {
      report.teacher_support = rankingSupport(
        report.data.map(({ name, index }) => ({
          index,
          contract: read(path.join(dataRoot, name, 'contract.json')),
          trajectory: read(path.join(dataRoot, name, 'trajectory.json')),
        })),
        config.runtime_teacher.minimum_informative_states ?? 16,
        config.runtime_teacher.minimum_source_scales ?? 2,
      );
      persist();
      if (config.runtime_teacher.require_support && !report.teacher_support.admitted)
        throw Error('Teacher support gate failed; raw data retained, learner not started');
    }
    const models = [{ name: 'initial', path: config.initial_model }];
    for (const seed of config.seeds) {
      const output = path.join(directory, 'training', String(seed));
      await run(
        'training-' + seed,
        trainer,
        [
          dataRoot,
          output,
          '--objective',
          objective,
          '--initialize',
          config.initial_model,
          '--steps',
          String(config.updates),
          '--batch',
          String(config.batch),
          '--gpu-memory-mib',
          String(config.gpu_memory_mib),
          '--seed',
          String(seed),
          '--minutes',
          '3',
        ],
        210000,
      );
      const latest = read(path.join(output, 'latest.json'));
      if (
        !latest.complete ||
        (objective === 'endpoint-ranking'
          ? latest.placement_policy_frozen !== false ||
            latest.trainable_scope !== 'endpoint-scorer' ||
            latest.rebuild_compatible !== false
          : !latest.placement_policy_frozen) ||
        !latest.optimizer_restored ||
        !latest.finite ||
        latest.model_sha256 !== hash(path.join(output, latest.model))
      )
        throw Error('Invalid rank checkpoint');
      const trained = { seed, latest };
      report.training.push(trained);
      persist();
      let model = path.join(output, latest.model);
      if (config.rank_fraction !== undefined) {
        const blended = path.join(output, 'ranking-blend.blzn');
        await run(
          'blend-' + seed,
          diagnostic,
          ['--blend-ranking', config.initial_model, model, String(config.rank_fraction), blended],
          30000,
        );
        model = blended;
        trained.evaluated_model = {
          path: model,
          sha256: hash(model),
          rank_fraction: config.rank_fraction,
        };
        persist();
      }
      models.push({ name: 'seed-' + seed, path: model });
    }
    for (const preserve of [true, false]) {
      const settings = {
        ...read(config.audit_settings),
        preserve_uv: preserve,
        gpu_memory_mib: config.gpu_memory_mib,
      };
      const settingsPath = path.join(directory, `audit-settings-uv${Number(preserve)}.json`);
      write(settingsPath, settings);
      for (const model of models) {
        const name = `audit-uv${Number(preserve)}-${model.name}`,
          output = path.join(directory, name + '.json');
        await run(
          name,
          diagnostic,
          ['--audit-model', config.audit_manifest, model.path, settingsPath, output],
          config.audit_minutes * 60000,
        );
        const audit = read(output);
        if (!audit.complete) throw Error('Incomplete quality diagnostic: ' + name);
        report.audits.push({
          preserve_uv: preserve,
          model: model.name,
          output,
          sha256: hash(output),
        });
        persist();
      }
    }
    report.comparisons = report.audits
      .filter((a) => a.model !== 'initial')
      .map((a) => ({
        preserve_uv: a.preserve_uv,
        model: a.model,
        ...comparePolicyRanking(
          read(
            report.audits.find((b) => b.preserve_uv === a.preserve_uv && b.model === 'initial')
              .output,
          ),
          read(a.output),
        ),
      }));
    report.complete = true;
  } catch (error) {
    report.error = String(error);
    throw error;
  } finally {
    report.finished_at = new Date().toISOString();
    persist();
  }
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 4) throw Error('policy-ranking.mjs CONFIG OUTPUT');
  await runPolicyRanking(process.argv[2], process.argv[3]);
}
