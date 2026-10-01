import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  exportBaselineCache,
  inspectBaselineCache,
  restoreBaselineCache,
  validateBaselineCache,
} from '../scripts/neural/baseline-cache.mjs';

const sha = (value) => createHash('sha256').update(value).digest('hex');
const read = (file) => JSON.parse(fs.readFileSync(file));
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value) + '\n');
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-baseline-cache-'));
  const checkout = path.join(root, 'source'),
    buildDirectory = path.join(checkout, 'build');
  fs.mkdirSync(buildDirectory, { recursive: true });
  const git = (...args) =>
    execFileSync('git', args, {
      cwd: checkout,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
  git('init', '-q');
  fs.writeFileSync(path.join(checkout, '.gitignore'), 'build/\n');
  fs.writeFileSync(path.join(checkout, 'source.cpp'), 'int main(){return 0;}\n');
  git('add', '.');
  git(
    '-c',
    'user.name=Cache test',
    '-c',
    'user.email=cache@test.invalid',
    'commit',
    '-qm',
    'Test-owned source',
  );
  const request = { baseline_revision: git('rev-parse', 'HEAD') };
  const binary = path.join(buildDirectory, 'blitz-neural-placement-prepare');
  const original = Buffer.from('test-owned native payload; never executed\n');
  fs.writeFileSync(binary, original, { mode: 0o700 });
  const evidenceFile = path.join(buildDirectory, 'baseline-build.json');
  const proof = {
    version: 1,
    original: { revision: request.baseline_revision, tree: git('rev-parse', 'HEAD^{tree}') },
    overlay: null,
    effective_tree: git('write-tree'),
    binary: { path: binary, sha256: sha(original) },
  };
  write(evidenceFile, proof);
  const environmentFile = path.join(root, 'environment.json');
  write(environmentFile, {
    cuda_architecture: 86,
    build_type: 'Release',
    toolkit: 'test-owned-toolkit',
  });
  const dependency = path.join(root, 'runtime-library'),
    header = path.join(root, 'sdk-header');
  fs.writeFileSync(dependency, 'runtime v1');
  fs.writeFileSync(header, 'header v1');
  const inspect = async () => ({
    config: read(environmentFile),
    runtime: sha(fs.readFileSync(dependency)),
    header: sha(fs.readFileSync(header)),
  });
  const options = {
    cacheDirectory: path.join(root, 'cache'),
    checkout,
    buildDirectory,
    evidenceFile,
    request,
    candidateRoot: checkout,
    inspect,
  };
  return {
    root,
    options,
    binary,
    original,
    proof,
    git,
    header,
    dependency,
    environmentFile,
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }),
  };
}

test('exported baseline restores exact bytes and fresh binary provenance; validation does not restore', async () => {
  const f = fixture();
  try {
    const exported = await exportBaselineCache(f.options);
    assert.equal(exported.status, 'exported');
    assert.match(exported.key, /^[a-f0-9]{64}$/);
    fs.unlinkSync(f.binary);
    write(f.options.evidenceFile, { ...f.proof, binary: undefined });
    const valid = await validateBaselineCache(f.options);
    assert.equal(valid.status, 'valid');
    assert.equal(fs.existsSync(f.binary), false);
    const hit = await restoreBaselineCache(f.options);
    assert.equal(hit.status, 'hit');
    assert.equal(hit.key, exported.key);
    assert.deepEqual(fs.readFileSync(f.binary), f.original);
    assert.equal(read(f.options.evidenceFile).binary.sha256, sha(f.original));
    assert.equal(read(f.options.evidenceFile).binary.path, f.binary);
    assert.equal(hit.fresh_runtime_validation_required, true);
    await assert.rejects(exportBaselineCache(f.options), /immutable/);
  } finally {
    f.cleanup();
  }
});

test('compile configuration, actual library and selected header changes each miss without replacing executable', async () => {
  for (const change of ['architecture', 'flags', 'toolkit', 'runtime', 'header']) {
    const f = fixture();
    try {
      await exportBaselineCache(f.options);
      fs.writeFileSync(f.binary, 'existing output must survive a miss');
      if (change === 'architecture')
        write(f.environmentFile, { ...read(f.environmentFile), cuda_architecture: 89 });
      if (change === 'flags')
        write(f.environmentFile, { ...read(f.environmentFile), build_type: 'Debug' });
      if (change === 'toolkit')
        write(f.environmentFile, { ...read(f.environmentFile), toolkit: 'another-toolkit' });
      if (change === 'runtime') fs.writeFileSync(f.dependency, 'runtime v2');
      if (change === 'header') fs.writeFileSync(f.header, 'header v2');
      const result = await restoreBaselineCache(f.options);
      assert.equal(result.status, 'miss', change);
      assert.match(result.reason, /environment changed/);
      assert.equal(fs.readFileSync(f.binary, 'utf8'), 'existing output must survive a miss');
    } finally {
      f.cleanup();
    }
  }
});

test('fresh source, indexed overlay and requested revision must agree independently of cache metadata', async () => {
  for (const change of ['unstaged', 'staged', 'untracked', 'proof', 'request']) {
    const f = fixture();
    try {
      await exportBaselineCache(f.options);
      if (['unstaged', 'staged'].includes(change))
        fs.writeFileSync(path.join(f.options.checkout, 'source.cpp'), 'int main(){return 1;}\n');
      if (change === 'staged') f.git('add', 'source.cpp');
      if (change === 'untracked')
        fs.writeFileSync(path.join(f.options.checkout, 'extra.hpp'), 'changed include');
      if (change === 'proof')
        write(f.options.evidenceFile, { ...f.proof, effective_tree: '0'.repeat(40) });
      if (change === 'request') f.options.request = { baseline_revision: '0'.repeat(40) };
      const result = await restoreBaselineCache(f.options);
      assert.equal(result.status, 'miss', change);
      assert.match(result.reason, /source\/overlay proof/);
    } finally {
      f.cleanup();
    }
  }
});

test('corruption and malformed paths produce visible build fallbacks with no partial restore', async () => {
  for (const change of [
    'missing',
    'json',
    'manifest',
    'payload',
    'path',
    'directory-link',
    'payload-link',
    'destination-link',
  ]) {
    const f = fixture();
    try {
      await exportBaselineCache(f.options);
      const manifest = path.join(f.options.cacheDirectory, 'manifest.json'),
        payload = path.join(f.options.cacheDirectory, 'binary.gz');
      if (change === 'missing') fs.rmSync(f.options.cacheDirectory, { recursive: true });
      if (change === 'json') fs.writeFileSync(manifest, '{');
      if (change === 'manifest') write(manifest, { ...read(manifest), key: '0'.repeat(64) });
      if (change === 'payload') fs.appendFileSync(payload, 'corrupt');
      if (change === 'path') {
        const m = read(manifest);
        m.binary.file = '../escape';
        write(manifest, m);
      }
      if (change === 'directory-link') {
        const actual = f.options.cacheDirectory + '-real';
        fs.renameSync(f.options.cacheDirectory, actual);
        fs.symlinkSync(actual, f.options.cacheDirectory);
      }
      if (change === 'payload-link') {
        const actual = payload + '-real';
        fs.renameSync(payload, actual);
        fs.symlinkSync(actual, payload);
      }
      if (change === 'destination-link') {
        fs.unlinkSync(f.binary);
        fs.symlinkSync(f.header, f.binary);
      }
      const result = await restoreBaselineCache(f.options);
      assert.equal(result.status, 'miss', change);
      assert.match(result.fallback, /build baseline/);
      assert.equal(fs.readFileSync(f.header, 'utf8'), 'header v1');
      assert.equal(
        fs.readdirSync(f.options.buildDirectory).some((n) => n.startsWith('.baseline-cache-')),
        false,
      );
    } finally {
      f.cleanup();
    }
  }
});

test('inspection failure is a bounded cache miss and does not erase previous output', async () => {
  const f = fixture();
  try {
    await exportBaselineCache(f.options);
    const result = await restoreBaselineCache({
      ...f.options,
      inspect: async () => {
        throw new Error('Fingerprint budget exceeded');
      },
    });
    assert.equal(result.status, 'miss');
    assert.match(result.reason, /Fingerprint budget exceeded/);
    assert.deepEqual(fs.readFileSync(f.binary), f.original);
    assert.deepEqual(read(f.options.evidenceFile), f.proof);
  } finally {
    f.cleanup();
  }
});

test('local admission verifies exact fixed files and requested source without inspecting or executing the host environment', async () => {
  const f = fixture();
  try {
    const exported = await exportBaselineCache(f.options);
    fs.unlinkSync(f.binary);
    const admitted = inspectBaselineCache(f.options);
    assert.equal(admitted.key, exported.key);
    assert.deepEqual(
      admitted.files.map((file) => file.path),
      ['manifest.json', 'binary.gz'],
    );
    for (const file of admitted.files) {
      const bytes = fs.readFileSync(path.join(f.options.cacheDirectory, file.path));
      assert.equal(file.sha256, sha(bytes));
      assert.equal(file.bytes, bytes.length);
    }
    assert.equal(admitted.fresh_environment_validation_required, true);
    assert.equal(fs.existsSync(f.binary), false);
    fs.writeFileSync(path.join(f.options.checkout, 'source.cpp'), 'int main(){return 2;}\n');
    f.git('add', 'source.cpp');
    f.git(
      '-c',
      'user.name=Cache test',
      '-c',
      'user.email=cache@test.invalid',
      'commit',
      '-qm',
      'Changed baseline',
    );
    assert.throws(
      () =>
        inspectBaselineCache({
          ...f.options,
          request: { baseline_revision: f.git('rev-parse', 'HEAD') },
        }),
      /original tree/,
    );
    fs.appendFileSync(path.join(f.options.cacheDirectory, 'binary.gz'), 'damage');
    assert.throws(() => inspectBaselineCache(f.options), /payload checksum/);
  } finally {
    f.cleanup();
  }
});

test('optional export CLI reports unavailable without failing a completed build', () => {
  const f = fixture();
  try {
    const request = path.join(f.root, 'request.json'),
      report = path.join(f.root, 'report.json');
    write(request, f.options.request);
    execFileSync(
      process.execPath,
      [
        fileURLToPath(new URL('../scripts/neural/baseline-cache.mjs', import.meta.url)),
        'export',
        f.options.cacheDirectory,
        f.options.checkout,
        f.options.buildDirectory,
        f.options.evidenceFile,
        request,
        report,
      ],
      { cwd: f.options.checkout, stdio: 'pipe' },
    );
    assert.equal(read(report).status, 'unavailable');
    assert.equal(read(report).baseline_build_unchanged, true);
    assert.deepEqual(fs.readFileSync(f.binary), f.original);
    assert.equal(fs.existsSync(f.options.cacheDirectory), false);
  } finally {
    f.cleanup();
  }
});
