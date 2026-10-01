import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import {
  cpuSetSize,
  detectBuildResources,
  planBuildResources,
} from '../scripts/neural/build-resources.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const GiB = 1024 ** 3;
const readFixture = (values) => (file) => {
  if (!(file in values)) throw Object.assign(new Error('not found'), { code: 'ENOENT' });
  return values[file];
};
const host = {
  '/proc/meminfo': `MemAvailable: ${(24 * GiB) / 1024} kB\n`,
  '/proc/self/cgroup': '0::/\n',
  '/sys/fs/cgroup/cpu.max': 'max 100000',
  '/sys/fs/cgroup/cpuset.cpus.effective': '0-15',
  '/sys/fs/cgroup/memory.max': 'max',
};

test('build jobs honor independent affinity, quota, cpuset and finite headroom', () => {
  for (const affinity of [2, 6]) {
    const plan = detectBuildResources({ affinity, read: readFixture(host) });
    assert.equal(plan.jobs, affinity);
    assert.equal(plan.cuda_jobs, Math.min(affinity, 2));
    assert.equal(plan.heavy_jobs, 1);
  }
  const quota = detectBuildResources({
    affinity: 16,
    read: readFixture({ ...host, '/sys/fs/cgroup/cpu.max': '275000 100000' }),
  });
  assert.equal(quota.jobs, 2); // floor fractional quota, independently of 16 CPU affinity
  const unlimitedMemory = detectBuildResources({
    affinity: 16,
    read: readFixture({
      ...host,
      '/sys/fs/cgroup/cpu.max': '275000 100000',
      '/sys/fs/cgroup/memory.max': '9223372036854771712',
    }),
  });
  assert.equal(unlimitedMemory.jobs, 2); // an unlimited memory sentinel must not discard the CPU quota
  const cpuset = detectBuildResources({
    affinity: 16,
    read: readFixture({ ...host, '/sys/fs/cgroup/cpuset.cpus.effective': '1,4-5' }),
  });
  assert.equal(cpuset.jobs, 3);
  assert.equal(planBuildResources({ affinity: 64, hostAvailableBytes: 128 * GiB }).jobs, 8);
});

test('nested cgroup v2 and host available RAM each constrain compilation', () => {
  const values = {
    ...host,
    '/proc/self/cgroup': '0::/parent/child',
    '/sys/fs/cgroup/parent/cpu.max': '180000 100000',
    '/sys/fs/cgroup/parent/child/cpu.max': '400000 100000',
  };
  assert.equal(detectBuildResources({ affinity: 16, read: readFixture(values) }).jobs, 1);
  delete values['/sys/fs/cgroup/parent/cpu.max'];
  values['/sys/fs/cgroup/memory.max'] = String(16 * GiB);
  values['/sys/fs/cgroup/memory.current'] = String(4 * GiB);
  let plan = detectBuildResources({ affinity: 16, read: readFixture(values) });
  assert.equal(plan.available_bytes, 12 * GiB);
  assert.equal(plan.jobs, 3);
  values['/proc/meminfo'] = `MemAvailable: ${(9 * GiB) / 1024} kB`;
  plan = detectBuildResources({ affinity: 16, read: readFixture(values) });
  assert.equal(plan.available_bytes, 9 * GiB);
  assert.equal(plan.jobs, 1);
  values['/sys/fs/cgroup/memory.current'] = String(17 * GiB);
  assert.equal(detectBuildResources({ affinity: 16, read: readFixture(values) }).jobs, 1);
});

test('v1 controller limits and unlimited memory sentinel are parsed safely', () => {
  const values = {
    '/proc/meminfo': host['/proc/meminfo'],
    '/proc/self/cgroup': '2:cpu,cpuacct:/tenant\n3:memory:/tenant\n4:cpuset:/tenant',
    '/sys/fs/cgroup/cpu,cpuacct/tenant/cpu.cfs_quota_us': '450000',
    '/sys/fs/cgroup/cpu,cpuacct/tenant/cpu.cfs_period_us': '100000',
    '/sys/fs/cgroup/cpuset/tenant/cpuset.cpus': '2-8',
    '/sys/fs/cgroup/memory/memory.limit_in_bytes': '9223372036854771712',
    '/sys/fs/cgroup/memory/tenant/memory.limit_in_bytes': String(14 * GiB),
    '/sys/fs/cgroup/memory/tenant/memory.usage_in_bytes': String(2 * GiB),
  };
  const plan = detectBuildResources({ affinity: 16, read: readFixture(values) });
  assert.equal(plan.jobs, 3);
  assert.equal(plan.cpu_limit, 4);
  assert.equal(plan.available_bytes, 12 * GiB);
  assert.deepEqual(plan.malformed, []);
});

test('unreadable or malformed constraints conservatively select one job', () => {
  for (const [name, value] of [
    ['/proc/meminfo', 'MemAvailable: unknown'],
    ['/sys/fs/cgroup/cpu.max', 'broken 100000'],
    ['/sys/fs/cgroup/cpu.max', 'max broken'],
    ['/sys/fs/cgroup/cpuset.cpus.effective', '0-,7'],
    ['/sys/fs/cgroup/memory.max', 'broken'],
  ]) {
    const plan = detectBuildResources({
      affinity: 16,
      read: readFixture({ ...host, [name]: value }),
    });
    assert.equal(plan.jobs, 1, name);
    assert.ok(plan.malformed.length);
  }
  assert.equal(detectBuildResources({ affinity: 16, read: readFixture({}) }).jobs, 1);
  for (const invalid of ['', '0-', '0-2,2-4', '1,,2', '4-2', '-1', '1.5'])
    assert.equal(cpuSetSize(invalid), null);
  assert.equal(cpuSetSize('0-3,7,10-11'), 7);
});

test('setup stage records failures without disabling errexit or continuing setup', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-setup-stages-'));
  try {
    for (const [name, operation, code] of [
      ['configure', 'blitz_stage configure bash -c "exit 7"', 7],
      ['function', 'fail_inside() { false; touch "$marker"; }; blitz_stage build fail_inside', 1],
      [
        'checksum',
        'blitz_stage_begin checksum; printf incorrect | sha256sum --check; blitz_stage_end 0',
        1,
      ],
      ['download', 'blitz_stage_begin download; exit 22', 22],
      ['terminated', 'blitz_stage_begin interrupted; kill -TERM $$', 143],
    ]) {
      const log = path.join(directory, name + '.jsonl'),
        marker = path.join(directory, name + '.continued');
      const result = spawnSync(
        'bash',
        ['-c', 'set -euo pipefail; source "$helper"; ' + operation + '; touch "$marker"'],
        {
          env: {
            ...process.env,
            helper: path.join(root, 'scripts/neural/setup-stages.sh'),
            BLITZ_SETUP_TIMINGS: log,
            marker,
          },
          encoding: 'utf8',
          timeout: 5000,
        },
      );
      assert.equal(result.status, code, result.stderr);
      assert.equal(fs.existsSync(marker), false);
      const rows = fs.readFileSync(log, 'utf8').trim().split('\n').map(JSON.parse);
      assert.equal(rows.length, 2);
      assert.equal(rows[0].event, 'start');
      assert.equal(rows[1].event, 'end');
      assert.equal(rows[1].exit_code, code);
      assert.equal(rows[1].stage, rows[0].stage);
      assert.equal(rows[1].elapsed_ms, rows[1].timestamp_ms - rows[0].timestamp_ms);
    }
    const successLog = path.join(directory, 'success.jsonl');
    const success = spawnSync(
      'bash',
      [
        '-c',
        'set -euo pipefail; source "$helper"; blitz_stage configure true; blitz_stage build true',
      ],
      {
        env: {
          ...process.env,
          helper: path.join(root, 'scripts/neural/setup-stages.sh'),
          BLITZ_SETUP_TIMINGS: successLog,
        },
        encoding: 'utf8',
        timeout: 5000,
      },
    );
    assert.equal(success.status, 0, success.stderr);
    assert.deepEqual(
      fs
        .readFileSync(successLog, 'utf8')
        .trim()
        .split('\n')
        .map(JSON.parse)
        .filter((row) => row.event === 'end')
        .map((row) => row.exit_code),
      [0, 0],
    );
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('cloud aggregate contains every native CTest executable and applies bounded target pools', (t) => {
  const cmake = process.env.CMAKE_COMMAND ?? 'cmake';
  if (
    spawnSync(cmake, ['--version']).status !== 0 ||
    spawnSync('ninja', ['--version']).status !== 0
  )
    return t.skip('CMake and Ninja required');
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-cloud-cmake-'));
  try {
    fs.symlinkSync(path.join(root, 'tests'), path.join(directory, 'tests'));
    fs.writeFileSync(path.join(directory, 'dummy.cpp'), 'int main() { return 0; }\n');
    fs.writeFileSync(path.join(directory, 'dummy.cu'), '// configure-only CUDA source\n');
    fs.writeFileSync(
      path.join(directory, 'CMakeLists.txt'),
      `cmake_minimum_required(VERSION 3.20)
project(CloudClosure LANGUAGES C CXX)
enable_testing()
set(BUILD_TESTING ON)
set(BLITZ_CUDA ON)
set(BLITZ_VULKAN ON)
set(BLITZ_NEURAL_TRAIN ON)
add_library(blitzremesher INTERFACE)
add_library(blitz_io INTERFACE)
add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
add_library(OpenSSL::Crypto INTERFACE IMPORTED)
add_library(CUDA::cudart INTERFACE IMPORTED)
add_library(CUDA::cublas INTERFACE IMPORTED)
foreach(tool blitz blitz-neural-train blitz-neural-action-train blitz-neural-diagnostics blitz-neural-cycle blitz-neural-placement-prepare unused-profiler)
  add_executable(\${tool} dummy.cpp)
endforeach()
add_library(cuda-runtime STATIC dummy.cpp dummy.cu)
set_source_files_properties(dummy.cu PROPERTIES HEADER_FILE_ONLY ON)
include("${root}/cmake/Tests.cmake")
`,
    );
    const build = path.join(directory, 'build');
    const configured = spawnSync(
      cmake,
      [
        '-S',
        directory,
        '-B',
        build,
        '-G',
        'Ninja',
        '-DCMAKE_PROJECT_INCLUDE=' + path.join(root, 'cmake/CloudBuild.cmake'),
        '-DBLITZ_CLOUD_JOBS=4',
        '-DBLITZ_CLOUD_CUDA_JOBS=2',
      ],
      { encoding: 'utf8', timeout: 30000 },
    );
    assert.equal(configured.status, 0, configured.stdout + configured.stderr);
    const manifest = JSON.parse(fs.readFileSync(path.join(build, 'cloud-build-targets.json')));
    const ctests = fs.readFileSync(path.join(build, 'CTestTestfile.cmake'), 'utf8');
    const nativeCommands = [...ctests.matchAll(/"([^"\n]+)"/g)]
      .map((match) => match[1])
      .filter((file) => file.startsWith(build + '/') && !file.endsWith('.json'));
    assert.ok(nativeCommands.length > 25);
    for (const file of nativeCommands)
      assert.ok(
        manifest.validation_targets.includes(path.basename(file)),
        'missing native CTest command ' + file,
      );
    for (const tool of [
      'blitz-neural-cycle',
      'blitz-neural-placement-prepare',
      'blitz-neural-diagnostics',
    ])
      assert.ok(manifest.validation_targets.includes(tool));
    assert.equal(manifest.validation_targets.includes('unused-profiler'), false);
    assert.equal(manifest.compile_pools['cuda-runtime'], 'blitz_cloud_cuda');
    assert.equal(manifest.compile_pools['blitz-neural-cycle'], 'blitz_cloud_heavy');
    assert.equal(manifest.compile_pools['blitz-neural-placement-prepare'], 'blitz_cloud_heavy');
    assert.equal(manifest.compile_pools['blitz-tests'], 'default');
    const graph = spawnSync(
      'ninja',
      ['-C', build, '-t', 'query', 'blitz-neural-cloud-validation'],
      { encoding: 'utf8' },
    );
    assert.equal(graph.status, 0, graph.stderr);
    for (const tool of manifest.validation_targets)
      assert.ok(graph.stdout.includes('\n    ' + tool + '\n'), 'aggregate graph misses ' + tool);
    assert.equal(graph.stdout.includes('unused-profiler'), false);
    const rules = fs.readFileSync(path.join(build, 'CMakeFiles/rules.ninja'), 'utf8');
    assert.match(rules, /pool blitz_cloud_cuda\n  depth = 2/);
    assert.match(rules, /pool blitz_cloud_heavy\n  depth = 1/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
