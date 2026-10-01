import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import {
  prepareBaselineOverlay,
  recordBaselineBinary,
  verifyBaselineBuild,
} from '../scripts/neural/baseline-overlay.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const write = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, value);
};
const json = (file, value) => write(file, JSON.stringify(value));
const git = (cwd, ...args) =>
  execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
function fixture(t, overlay = 'worker-join-v1') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-baseline-overlay-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const checkout = root + '/source',
    candidateRoot = root + '/candidate';
  fs.mkdirSync(checkout);
  git(checkout, 'init');
  const original = { 'training/workers.hpp': 'original teacher body\n' };
  if (overlay !== 'worker-join-v1')
    original['training/placement_teacher.hpp'] = 'original seed admission\n';
  if (overlay === 'worker-join-correctness-v3') {
    original['src/neural/action.cpp'] = 'original action executor and audit predicate\n';
    original['src/neural/action_gpu.cu'] = 'original GPU placement admission\n';
    original['src/neural/generate.cpp'] = 'original final confirmation\n';
  }
  for (const [file, data] of Object.entries(original)) write(checkout + '/' + file, data);
  write(checkout + '/unrelated.txt', 'unchanged\n');
  git(checkout, 'add', '.');
  git(
    checkout,
    '-c',
    'user.name=Fixture',
    '-c',
    'user.email=fixture@example.invalid',
    'commit',
    '-m',
    'Test-owned baseline',
  );
  const revision = git(checkout, 'rev-parse', 'HEAD'),
    originalTree = git(checkout, 'rev-parse', 'HEAD^{tree}');
  const contents = {
    'training/workers.hpp': 'original teacher body\njoin retirement\n',
    'training/worker_retirement.hpp': 'shared retirement helper\n',
    'src/neural/teardown_trace.hpp': 'shared trace helper\n',
  };
  if (overlay !== 'worker-join-v1') {
    contents['training/placement_teacher.hpp'] = 'corrected seed admission\n';
    contents['training/teacher_seed.hpp'] = 'shared source adjacent destination admission\n';
  }
  if (overlay === 'worker-join-correctness-v3') {
    contents['src/neural/action.cpp'] = 'corrected action executor and audit predicate\n';
    contents['src/neural/action_gpu.cu'] = 'corrected GPU placement admission\n';
    contents['src/neural/generate.cpp'] = 'corrected final confirmation\n';
    contents['src/neural/audit_measurement.hpp'] = 'shared measurement validity predicate\n';
  }
  for (const [file, data] of Object.entries(contents)) {
    write(checkout + '/' + file, data);
    write(candidateRoot + '/' + file, data);
  }
  git(checkout, 'add', '.');
  const patchFile = root + '/' + overlay + '.patch';
  fs.writeFileSync(
    patchFile,
    execFileSync('git', ['diff', '--cached', '--binary', '--full-index'], { cwd: checkout }),
  );
  const definition = {
    version: 1,
    id: overlay,
    baseline_revision: revision,
    original_tree: originalTree,
    modified_tree: git(checkout, 'write-tree'),
    patch_sha256: hash(fs.readFileSync(patchFile)),
    files: Object.entries(contents).map(([file, data]) => ({
      path: file,
      before_sha256: Object.hasOwn(original, file) ? hash(original[file]) : null,
      after_sha256: hash(data),
    })),
  };
  const definitionPath = root + '/' + overlay + '.json';
  json(definitionPath, definition);
  git(checkout, 'reset', '--hard', 'HEAD');
  return {
    root,
    checkout,
    candidateRoot,
    definition,
    definitionPath,
    patchFile,
    request: { baseline_revision: revision, baseline_overlay: overlay },
    output: root + '/build.json',
  };
}

test('baseline overlay records original and modified source identities and verified executable', (t) => {
  const f = fixture(t),
    prepared = prepareBaselineOverlay(f);
  assert.equal(prepared.original.revision, f.request.baseline_revision);
  assert.equal(prepared.original.tree, f.definition.original_tree);
  assert.equal(prepared.effective_tree, f.definition.modified_tree);
  assert.deepEqual(prepared.overlay.files, f.definition.files);
  assert.match(prepared.label, /join-retirement overlay worker-join-v1/);
  assert.equal(git(f.checkout, 'rev-parse', 'HEAD'), f.request.baseline_revision);
  const binary = f.root + '/native-binary';
  fs.writeFileSync(binary, 'compiled baseline identity');
  const built = recordBaselineBinary(f.output, binary);
  assert.equal(built.binary.sha256, hash('compiled baseline identity'));
  assert.deepEqual(verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }), built);
  fs.appendFileSync(binary, 'changed');
  assert.throws(
    () => verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }),
    /executable checksum/,
  );
});

test('unmodified baseline remains explicitly separate from a requested overlay', (t) => {
  const f = fixture(t);
  delete f.request.baseline_overlay;
  const prepared = prepareBaselineOverlay(f);
  assert.equal(prepared.overlay, null);
  assert.equal(prepared.effective_tree, prepared.original.tree);
  assert.equal(git(f.checkout, 'status', '--porcelain'), '');
  const binary = f.root + '/native-binary';
  fs.writeFileSync(binary, 'original binary');
  recordBaselineBinary(f.output, binary);
  assert.equal(verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }).overlay, null);
  assert.throws(
    () =>
      verifyBaselineBuild({
        ...f,
        evidenceFile: f.output,
        binary,
        request: { ...f.request, baseline_overlay: 'worker-join-v1' },
      }),
    /overlay evidence/,
  );
});

for (const overlay of ['worker-join-v1', 'worker-join-seed-v2', 'worker-join-correctness-v3'])
  for (const failure of [
    'dirty',
    'source-revision',
    'input-hash',
    'patch-hash',
    'context',
    'extra-file',
    'candidate-helper',
    'result-tree',
  ])
    test(overlay + ' rejects ' + failure + ' without publishing build provenance', (t) => {
      const f = fixture(t, overlay);
      if (failure === 'dirty') write(f.checkout + '/unrelated.txt', 'modified');
      if (failure === 'source-revision') f.request.baseline_revision = '0'.repeat(40);
      if (failure === 'input-hash') f.definition.files[0].before_sha256 = '0'.repeat(64);
      if (failure === 'patch-hash') fs.appendFileSync(f.patchFile, '\n');
      if (failure === 'context') {
        const patch = fs
          .readFileSync(f.patchFile, 'utf8')
          .replace(' original teacher body\n', ' wrong source context\n');
        fs.writeFileSync(f.patchFile, patch);
        f.definition.patch_sha256 = hash(patch);
      }
      if (failure === 'extra-file') {
        fs.appendFileSync(
          f.patchFile,
          '\ndiff --git a/unrelated.txt b/unrelated.txt\n--- a/unrelated.txt\n+++ b/unrelated.txt\n@@ -1 +1 @@\n-unchanged\n+unauthorized\n',
        );
        f.definition.patch_sha256 = hash(fs.readFileSync(f.patchFile));
      }
      if (failure === 'candidate-helper')
        write(f.candidateRoot + '/training/worker_retirement.hpp', 'different helper');
      if (failure === 'result-tree') f.definition.modified_tree = '0'.repeat(40);
      json(f.definitionPath, f.definition);
      assert.throws(
        () => prepareBaselineOverlay(f),
        failure === 'extra-file' ? /allowlist/ : undefined,
      );
      assert.equal(fs.existsSync(f.output), false);
      if (failure !== 'dirty' && failure !== 'result-tree')
        assert.equal(git(f.checkout, 'status', '--porcelain'), '');
    });

test('verified executable cannot authenticate changed overlay metadata or another candidate helper', (t) => {
  const f = fixture(t);
  prepareBaselineOverlay(f);
  const binary = f.root + '/binary';
  fs.writeFileSync(binary, 'binary');
  const built = recordBaselineBinary(f.output, binary);
  built.overlay.patch_sha256 = '0'.repeat(64);
  json(f.output, built);
  assert.throws(
    () => verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }),
    /overlay evidence/,
  );
  built.overlay.patch_sha256 = f.definition.patch_sha256;
  json(f.output, built);
  built.original.revision = '0'.repeat(40);
  json(f.output, built);
  assert.throws(
    () =>
      verifyBaselineBuild({
        ...f,
        evidenceFile: f.output,
        binary,
        request: { ...f.request, baseline_revision: built.original.revision },
      }),
    /overlay evidence/,
  );
  built.original.revision = f.request.baseline_revision;
  json(f.output, built);
  write(f.candidateRoot + '/src/neural/teardown_trace.hpp', 'changed trace');
  assert.throws(
    () => verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }),
    /shared header mismatch/,
  );
});

test('seed-admission overlay retains its own identity and verifies the common gate helper', (t) => {
  const f = fixture(t, 'worker-join-seed-v2');
  // Optimized and historical teachers intentionally differ outside the shared
  // gate helper; both applied source files still have exact recorded hashes.
  write(
    f.candidateRoot + '/training/placement_teacher.hpp',
    'optimized teacher with shared gate\n',
  );
  const prepared = prepareBaselineOverlay(f);
  assert.equal(prepared.overlay.id, 'worker-join-seed-v2');
  assert.match(prepared.label, /join-retirement and seed-admission overlay worker-join-seed-v2/);
  assert.equal(prepared.overlay.files.length, 5);
  const binary = f.root + '/binary';
  fs.writeFileSync(binary, 'corrected historical baseline');
  recordBaselineBinary(f.output, binary);
  assert.equal(
    verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }).effective_tree,
    f.definition.modified_tree,
  );
  write(f.candidateRoot + '/training/teacher_seed.hpp', 'different admission semantics\n');
  assert.throws(
    () => verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }),
    /shared header mismatch: training\/teacher_seed.hpp/,
  );
});

test('seed gate mismatch and renamed lifecycle-only definitions cannot provision the new overlay', (t) => {
  const mismatch = fixture(t, 'worker-join-seed-v2');
  write(mismatch.candidateRoot + '/training/teacher_seed.hpp', 'different admission\n');
  assert.throws(() => prepareBaselineOverlay(mismatch), /shared header mismatch/);
  assert.equal(git(mismatch.checkout, 'status', '--porcelain'), '');
  assert.equal(fs.existsSync(mismatch.output), false);
  const renamed = fixture(t);
  renamed.request.baseline_overlay = 'worker-join-seed-v2';
  renamed.definition.id = 'worker-join-seed-v2';
  json(renamed.definitionPath, renamed.definition);
  assert.throws(() => prepareBaselineOverlay(renamed), /Invalid fixed baseline/);
  assert.equal(git(renamed.checkout, 'status', '--porcelain'), '');
  assert.equal(fs.existsSync(renamed.output), false);
});

test('native-correctness overlay preserves serial implementation freedom but pins all shared predicates', (t) => {
  const f = fixture(t, 'worker-join-correctness-v3');
  for (const file of [
    'training/placement_teacher.hpp',
    'src/neural/action.cpp',
    'src/neural/action_gpu.cu',
    'src/neural/generate.cpp',
  ])
    write(
      f.candidateRoot + '/' + file,
      'optimized implementation with the same correctness contract\n',
    );
  const prepared = prepareBaselineOverlay(f);
  assert.equal(prepared.overlay.id, 'worker-join-correctness-v3');
  assert.match(prepared.label, /join-retirement and native-correctness overlay/);
  assert.equal(prepared.overlay.files.length, 9);
  const binary = f.root + '/binary';
  fs.writeFileSync(binary, 'corrected historical serial baseline');
  recordBaselineBinary(f.output, binary);
  for (const file of [
    'src/neural/audit_measurement.hpp',
    'src/neural/teardown_trace.hpp',
    'training/teacher_seed.hpp',
    'training/worker_retirement.hpp',
  ]) {
    const target = f.candidateRoot + '/' + file;
    const original = fs.readFileSync(target);
    assert.equal(
      verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }).effective_tree,
      f.definition.modified_tree,
    );
    fs.appendFileSync(target, 'different correctness contract\n');
    assert.throws(
      () => verifyBaselineBuild({ ...f, evidenceFile: f.output, binary }),
      /shared header mismatch/,
      file,
    );
    fs.writeFileSync(target, original);
  }
});

test('legacy seed overlay cannot be renamed into the complete native-correctness allowlist', (t) => {
  const f = fixture(t, 'worker-join-seed-v2');
  f.request.baseline_overlay = 'worker-join-correctness-v3';
  f.definition.id = 'worker-join-correctness-v3';
  json(f.definitionPath, f.definition);
  assert.throws(() => prepareBaselineOverlay(f), /Invalid fixed baseline/);
  assert.equal(git(f.checkout, 'status', '--porcelain'), '');
  assert.equal(fs.existsSync(f.output), false);
});

test('unknown overlay identifiers cannot select arbitrary definition files', (t) => {
  const f = fixture(t);
  for (const id of ['../worker-join-v1', '__proto__', 'unreviewed']) {
    f.request.baseline_overlay = id;
    assert.throws(() => prepareBaselineOverlay(f), /Unknown fixed baseline overlay/);
  }
  assert.equal(git(f.checkout, 'status', '--porcelain'), '');
});
