// One reviewed lifecycle-only overlay for the frozen teacher benchmark baseline.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

const definitionFile = fileURLToPath(new URL('./baselines/worker-join-v1.json', import.meta.url));
const allowed = [
  'src/neural/teardown_trace.hpp',
  'training/worker_retirement.hpp',
  'training/workers.hpp',
];
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fileHash = (file) => hash(fs.readFileSync(file));
const read = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n');
const git = (cwd, ...args) =>
  execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();

function definition(file) {
  const value = read(file);
  if (
    value.version !== 1 ||
    value.id !== 'worker-join-v1' ||
    !isDeepStrictEqual(value.files?.map((f) => f.path).sort(), allowed) ||
    ![value.baseline_revision, value.original_tree, value.modified_tree].every((s) =>
      /^[a-f0-9]{40}$/.test(s),
    ) ||
    !/^[a-f0-9]{64}$/.test(value.patch_sha256) ||
    value.files.some(
      (f) =>
        (f.before_sha256 !== null && !/^[a-f0-9]{64}$/.test(f.before_sha256)) ||
        !/^[a-f0-9]{64}$/.test(f.after_sha256),
    )
  )
    throw new Error('Invalid fixed baseline overlay definition');
  return value;
}

function candidateHeaders(value, candidateRoot) {
  for (const file of value.files.filter((f) => f.path !== 'training/workers.hpp'))
    if (fileHash(path.join(candidateRoot, file.path)) !== file.after_sha256)
      throw new Error('Baseline/candidate retirement header mismatch: ' + file.path);
}

export function prepareBaselineOverlay({
  checkout,
  request,
  candidateRoot,
  output,
  definitionPath = definitionFile,
}) {
  if (git(checkout, 'status', '--porcelain', '--untracked-files=all'))
    throw new Error('Baseline overlay needs a clean source checkout');
  const original = {
    revision: git(checkout, 'rev-parse', 'HEAD'),
    tree: git(checkout, 'rev-parse', 'HEAD^{tree}'),
  };
  if (original.revision !== request.baseline_revision)
    throw new Error('Baseline source revision mismatch');
  const evidence = {
    version: 1,
    original,
    overlay: null,
    effective_tree: original.tree,
    label: original.revision,
  };
  if (request.baseline_overlay !== undefined) {
    const value = definition(definitionPath);
    if (
      request.baseline_overlay !== value.id ||
      original.revision !== value.baseline_revision ||
      original.tree !== value.original_tree
    )
      throw new Error('Baseline overlay source identity mismatch');
    const patchFile = path.join(path.dirname(definitionPath), value.id + '.patch');
    if (fileHash(patchFile) !== value.patch_sha256)
      throw new Error('Baseline overlay patch checksum mismatch');
    const edited = git(checkout, 'apply', '--numstat', '-z', patchFile)
      .split('\0')
      .filter(Boolean)
      .map((row) => row.split('\t')[2])
      .sort();
    if (!isDeepStrictEqual(edited, allowed))
      throw new Error('Baseline overlay changes files outside its three-file allowlist');
    for (const file of value.files) {
      const source = path.join(checkout, file.path),
        before = fs.existsSync(source) ? fileHash(source) : null;
      if (before !== file.before_sha256)
        throw new Error('Baseline overlay input checksum mismatch: ' + file.path);
    }
    candidateHeaders(value, candidateRoot);
    git(checkout, 'apply', '--check', '--index', patchFile);
    git(checkout, 'apply', '--index', patchFile);
    if (
      git(checkout, 'diff', '--name-only') ||
      !isDeepStrictEqual(
        git(checkout, 'diff', '--cached', '--name-only').split('\n').sort(),
        allowed,
      )
    )
      throw new Error('Unexpected baseline source changes after overlay');
    for (const file of value.files)
      if (fileHash(path.join(checkout, file.path)) !== file.after_sha256)
        throw new Error('Baseline overlay output checksum mismatch: ' + file.path);
    evidence.effective_tree = git(checkout, 'write-tree');
    if (evidence.effective_tree !== value.modified_tree)
      throw new Error('Baseline overlay result tree mismatch');
    evidence.overlay = value;
    evidence.label += ' + join-retirement overlay ' + value.id;
  }
  write(output, evidence);
  return evidence;
}

export function recordBaselineBinary(evidenceFile, binary) {
  const evidence = read(evidenceFile);
  evidence.binary = { path: path.resolve(binary), sha256: fileHash(binary) };
  write(evidenceFile, evidence);
  return evidence;
}

export function verifyBaselineBuild({
  evidenceFile,
  binary,
  request,
  candidateRoot,
  definitionPath = definitionFile,
}) {
  const evidence = read(evidenceFile);
  if (
    evidence.version !== 1 ||
    evidence.original?.revision !== request.baseline_revision ||
    evidence.binary?.path !== path.resolve(binary) ||
    evidence.binary.sha256 !== fileHash(binary)
  )
    throw new Error('Baseline build identity or executable checksum mismatch');
  if (request.baseline_overlay !== undefined) {
    const value = definition(definitionPath);
    if (
      request.baseline_overlay !== value.id ||
      request.baseline_revision !== value.baseline_revision ||
      !isDeepStrictEqual(evidence.overlay, value) ||
      evidence.original.tree !== value.original_tree ||
      evidence.effective_tree !== value.modified_tree
    )
      throw new Error('Baseline lifecycle overlay evidence mismatch');
    candidateHeaders(value, candidateRoot);
  } else if (evidence.overlay !== null || evidence.effective_tree !== evidence.original.tree)
    throw new Error('Unrequested baseline source overlay');
  return evidence;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [action, checkoutOrEvidence, requestOrBinary, output] = process.argv.slice(2);
  if (action === 'prepare')
    prepareBaselineOverlay({
      checkout: checkoutOrEvidence,
      request: read(requestOrBinary),
      candidateRoot: process.cwd(),
      output,
    });
  else if (action === 'record') recordBaselineBinary(checkoutOrEvidence, requestOrBinary);
  else
    throw new Error(
      'Usage: baseline-overlay.mjs prepare CHECKOUT REQUEST OUTPUT | record EVIDENCE BINARY',
    );
}
