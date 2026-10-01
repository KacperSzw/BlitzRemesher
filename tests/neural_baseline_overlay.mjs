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
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-baseline-overlay-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const checkout = root + '/source',
    candidateRoot = root + '/candidate';
  fs.mkdirSync(checkout);
  git(checkout, 'init');
  write(checkout + '/training/workers.hpp', 'original teacher body\n');
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
  for (const [file, data] of Object.entries(contents)) {
    write(checkout + '/' + file, data);
    write(candidateRoot + '/' + file, data);
  }
  git(checkout, 'add', '.');
  const patchFile = root + '/worker-join-v1.patch';
  fs.writeFileSync(
    patchFile,
    execFileSync('git', ['diff', '--cached', '--binary', '--full-index'], { cwd: checkout }),
  );
  const definition = {
    version: 1,
    id: 'worker-join-v1',
    baseline_revision: revision,
    original_tree: originalTree,
    modified_tree: git(checkout, 'write-tree'),
    patch_sha256: hash(fs.readFileSync(patchFile)),
    files: Object.entries(contents).map(([file, data]) => ({
      path: file,
      before_sha256: file === 'training/workers.hpp' ? hash('original teacher body\n') : null,
      after_sha256: hash(data),
    })),
  };
  const definitionPath = root + '/worker-join-v1.json';
  json(definitionPath, definition);
  git(checkout, 'reset', '--hard', 'HEAD');
  return {
    root,
    checkout,
    candidateRoot,
    definition,
    definitionPath,
    patchFile,
    request: { baseline_revision: revision, baseline_overlay: 'worker-join-v1' },
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
  test('baseline overlay rejects ' + failure + ' without publishing build provenance', (t) => {
    const f = fixture(t);
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
    /retirement header mismatch/,
  );
});
