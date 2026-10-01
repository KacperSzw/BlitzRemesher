// Reuse only a verified Ubuntu-built baseline executable, never a build tree.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { gzipSync, gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { IMAGE } from './runpod-api.mjs';
import { read, write } from './artifacts.mjs';
import { recordBaselineBinary, verifyBaselineBuild } from './baseline-overlay.mjs';

const executable = 'blitz-neural-placement-prepare';
const maximumBinary = 64 * 1024 * 1024;
const maximumManifest = 2 * 1024 * 1024;
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const canonical = (value) =>
  JSON.stringify(value, (_, item) =>
    item && typeof item === 'object' && !Array.isArray(item)
      ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
      : item,
  );
const command = (program, args, cwd, timeout = 30000) =>
  execFileSync(program, args, {
    cwd,
    encoding: 'utf8',
    timeout,
    maxBuffer: 32 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
const git = (checkout, ...args) => command('git', args, checkout);

function regular(file, maximum = Infinity) {
  const info = fs.lstatSync(file);
  if (!info.isFile() || info.size > maximum || fs.realpathSync(file) !== path.resolve(file))
    throw new Error('Cache requires a bounded regular file without symlink components: ' + file);
  return info;
}
function cacheDirectory(directory) {
  if (
    !fs.lstatSync(directory).isDirectory() ||
    fs.realpathSync(directory) !== path.resolve(directory)
  )
    throw new Error('Cache directory must not contain symlinks');
}
async function fileIdentity(file, deadline = Infinity) {
  const actual = fs.realpathSync(file),
    hash = createHash('sha256');
  const info = fs.statSync(actual);
  if (!info.isFile()) throw new Error('Dependency is not a regular file: ' + file);
  for await (const bytes of fs.createReadStream(actual)) {
    if (Date.now() >= deadline) throw new Error('Baseline dependency fingerprint deadline reached');
    hash.update(bytes);
  }
  return { path: actual, bytes: info.size, sha256: hash.digest('hex') };
}
function sourceProof({ checkout, evidenceFile, request }) {
  const evidence = read(evidenceFile);
  if (
    evidence.version !== 1 ||
    evidence.original?.revision !== request.baseline_revision ||
    git(checkout, 'rev-parse', 'HEAD') !== evidence.original.revision ||
    git(checkout, 'rev-parse', 'HEAD^{tree}') !== evidence.original.tree ||
    git(checkout, 'diff', '--name-only') ||
    git(checkout, 'ls-files', '--others', '--exclude-standard') ||
    git(checkout, 'write-tree') !== evidence.effective_tree
  )
    throw new Error('Fresh baseline source/overlay proof does not match its checkout');
  return {
    original: evidence.original,
    overlay: evidence.overlay,
    effective_tree: evidence.effective_tree,
  };
}

// Versions identify package-managed headers; explicitly hash the selected public
// SDK headers and actual linked libraries. Do not scan whole CUDA/Torch trees.
export async function inspectBaselineEnvironment({ buildDirectory, binary }) {
  const started = Date.now();
  const deadline = started + 90000;
  const run = (program, args, cwd) => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error('Baseline dependency fingerprint deadline reached');
    return command(program, args, cwd, Math.min(30000, remaining));
  };
  const cache = Object.fromEntries(
    fs
      .readFileSync(path.join(buildDirectory, 'CMakeCache.txt'), 'utf8')
      .split('\n')
      .flatMap((line) => {
        const match = /^([^/#][^:]*):[^=]+=(.*)$/.exec(line);
        return match ? [[match[1], match[2]]] : [];
      }),
  );
  const os = fs.readFileSync('/etc/os-release', 'utf8');
  if (!/^ID=ubuntu$/m.test(os) || !/^VERSION_ID="24\.04"$/m.test(os) || process.arch !== 'x64')
    throw new Error('Baseline cache supports the pinned Ubuntu 24.04 x86_64 container only');
  if (
    cache.CMAKE_BUILD_TYPE !== 'Release' ||
    cache.BLITZ_CUDA !== 'ON' ||
    cache.BLITZ_VULKAN !== 'ON' ||
    cache.BLITZ_NEURAL_TRAIN !== 'OFF' ||
    cache.BLITZ_ACQUISITION !== 'OFF' ||
    !/^\d+(?:-(?:real|virtual))?(;\d+(?:-(?:real|virtual))?)*$/.test(
      cache.CMAKE_CUDA_ARCHITECTURES ?? '',
    )
  )
    throw new Error('Unsupported baseline cache build configuration');
  const commands = run(cache.CMAKE_MAKE_PROGRAM, ['-t', 'commands', executable], buildDirectory);
  if (/-m(?:arch|tune)=native/.test(commands))
    throw new Error('Host-native CPU tuning is not portable');
  const interpreter = run('readelf', ['-l', binary]).match(
    /Requesting program interpreter: ([^\]]+)/,
  )?.[1];
  if (interpreter !== '/lib64/ld-linux-x86-64.so.2')
    throw new Error('Non-Ubuntu executable interpreter');
  const dynamic = run('readelf', ['-d', binary]);
  if (dynamic.includes('/nix/store/') || dynamic.includes('libtorch'))
    throw new Error('Baseline executable contains unsupported host/training dependencies');
  const tools = {};
  for (const key of [
    'CMAKE_C_COMPILER',
    'CMAKE_CXX_COMPILER',
    'CMAKE_CUDA_COMPILER',
    'CMAKE_AR',
    'CMAKE_LINKER',
    'CMAKE_MAKE_PROGRAM',
    'CMAKE_COMMAND',
    'Vulkan_GLSLANG_VALIDATOR_EXECUTABLE',
  ]) {
    const file = cache[key];
    if (!file || file.endsWith('-NOTFOUND')) throw new Error('Missing build tool: ' + key);
    tools[key] = { ...(await fileIdentity(file, deadline)), version: run(file, ['--version']) };
  }
  const cuda = path.resolve(path.dirname(cache.CMAKE_CUDA_COMPILER), '../include');
  const headers = [
    path.join(cuda, 'cuda.h'),
    path.join(cuda, 'cuda_runtime_api.h'),
    path.join(cuda, 'cublas_v2.h'),
    path.join(cuda, 'npp.h'),
    path.join(cache.Vulkan_INCLUDE_DIR, 'vulkan/vulkan_core.h'),
    path.join(cache.OPENSSL_INCLUDE_DIR, 'openssl/opensslv.h'),
    '/usr/include/nlohmann/json.hpp',
  ];
  const headerIdentities = [];
  for (const header of headers) headerIdentities.push(await fileIdentity(header, deadline));
  const linked = run('ldd', [binary]);
  if (/not found/.test(linked)) throw new Error('Baseline runtime dependency is unresolved');
  const libraries = new Set([interpreter]);
  for (const line of linked.split('\n')) {
    const resolved = line.match(/=>\s+(\/\S+)/)?.[1] ?? line.trim().match(/^(\/\S+)/)?.[1];
    if (resolved) libraries.add(resolved);
  }
  if (libraries.size < 2 || libraries.size > 64)
    throw new Error('Invalid runtime dependency closure');
  const runtime = [];
  let bytes = 0;
  for (const file of [...libraries].sort()) {
    // NVIDIA host driver state is verified on every deployment, not cached.
    if (/\/lib(?:cuda|nvidia[^/]*|GLX_nvidia|EGL_nvidia)[.]so/.test(file)) continue;
    bytes += fs.statSync(file).size;
    if (bytes > 2 * 1024 * 1024 * 1024 || Date.now() - started > 90000)
      throw new Error('Baseline dependency fingerprint exceeds its bounded budget');
    runtime.push(await fileIdentity(file, deadline));
  }
  // These affect scheduling only. The expanded target commands remain strict,
  // so a future include that changes code generation still invalidates reuse.
  for (const key of ['BLITZ_CLOUD_JOBS', 'BLITZ_CLOUD_CUDA_JOBS', 'CMAKE_PROJECT_INCLUDE'])
    delete cache[key];
  return {
    container_image: IMAGE,
    architecture: process.arch,
    os_release_sha256: digest(os),
    packages: run('dpkg-query', ['-W', '-f=${binary:Package}\t${Version}\t${Architecture}\n'])
      .split('\n')
      .sort(),
    cmake_cache: cache,
    commands_sha256: digest(commands),
    tools,
    headers: headerIdentities,
    runtime_libraries: runtime,
    runtime_driver_excluded: true,
  };
}

function optionsPaths(options) {
  const buildDirectory = path.resolve(options.buildDirectory);
  return {
    ...options,
    checkout: path.resolve(options.checkout),
    buildDirectory,
    cacheDirectory: path.resolve(options.cacheDirectory),
    evidenceFile: path.resolve(options.evidenceFile),
    candidateRoot: path.resolve(options.candidateRoot ?? process.cwd()),
    binary: path.join(buildDirectory, executable),
  };
}
function verifyProof(options, binary, proof, sha256, directory) {
  const temporary = path.join(directory, 'proof.json');
  write(temporary, { version: 1, ...proof, binary: { path: binary, sha256 } });
  verifyBaselineBuild({
    evidenceFile: temporary,
    binary,
    request: options.request,
    candidateRoot: options.candidateRoot,
    definitionPath: options.definitionPath,
  });
}

function cachePayload(directory) {
  cacheDirectory(directory);
  const manifestFile = path.join(directory, 'manifest.json');
  const manifestInfo = regular(manifestFile, maximumManifest);
  const manifest = read(manifestFile);
  if (
    manifest.version !== 1 ||
    manifest.kind !== 'ubuntu-baseline-executable' ||
    manifest.binary?.name !== executable ||
    manifest.binary?.file !== 'binary.gz' ||
    !Number.isSafeInteger(manifest.binary.bytes) ||
    manifest.binary.bytes <= 0 ||
    manifest.binary.bytes > maximumBinary ||
    !/^[a-f0-9]{64}$/.test(manifest.key ?? '') ||
    manifest.key !==
      digest(
        canonical({
          version: 1,
          source: manifest.source,
          environment: manifest.environment,
          binary: manifest.binary,
        }),
      )
  )
    throw new Error('Malformed baseline cache manifest');
  const payload = path.join(directory, 'binary.gz');
  const payloadInfo = regular(payload, maximumBinary);
  const compressed = fs.readFileSync(payload);
  if (
    compressed.length !== manifest.binary.compressed_bytes ||
    digest(compressed) !== manifest.binary.compressed_sha256
  )
    throw new Error('Baseline cache compressed payload checksum mismatch');
  const bytes = gunzipSync(compressed, { maxOutputLength: maximumBinary });
  if (bytes.length !== manifest.binary.bytes || digest(bytes) !== manifest.binary.sha256)
    throw new Error('Baseline cache executable checksum mismatch');
  return {
    manifest,
    bytes,
    files: [
      {
        path: 'manifest.json',
        bytes: manifestInfo.size,
        sha256: digest(fs.readFileSync(manifestFile)),
      },
      { path: 'binary.gz', bytes: payloadInfo.size, sha256: digest(compressed) },
    ],
  };
}

// Local bundle admission checks bytes and source identity only. It deliberately
// cannot declare an environment hit on a Nix host; restore checks that remotely.
export function inspectBaselineCache({
  cacheDirectory: directory,
  request,
  candidateRoot = process.cwd(),
  definitionPath,
}) {
  const { manifest, bytes, files } = cachePayload(path.resolve(directory));
  if (
    git(candidateRoot, 'rev-parse', request.baseline_revision + '^{tree}') !==
    manifest.source?.original?.tree
  )
    throw new Error('Cached baseline original tree differs from requested revision');
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'blitz-cache-admission-'));
  try {
    const binary = path.join(temporary, executable);
    fs.writeFileSync(binary, bytes, { flag: 'wx', mode: 0o600 });
    verifyProof(
      { request, candidateRoot, definitionPath },
      binary,
      manifest.source,
      manifest.binary.sha256,
      temporary,
    );
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
  return {
    key: manifest.key,
    files,
    source: manifest.source,
    binary_sha256: manifest.binary.sha256,
    fresh_environment_validation_required: true,
  };
}

export async function exportBaselineCache(input) {
  const options = optionsPaths(input),
    inspect = input.inspect ?? inspectBaselineEnvironment;
  if (fs.existsSync(options.cacheDirectory))
    throw new Error('Choose a new immutable baseline cache directory');
  regular(options.binary, maximumBinary);
  verifyBaselineBuild({ ...options });
  const proof = sourceProof(options);
  const environment = await inspect(options);
  const bytes = fs.readFileSync(options.binary),
    compressed = gzipSync(bytes, { level: 1 });
  const binary = {
    file: 'binary.gz',
    name: executable,
    bytes: bytes.length,
    sha256: digest(bytes),
    compressed_bytes: compressed.length,
    compressed_sha256: digest(compressed),
  };
  const key = digest(canonical({ version: 1, source: proof, environment, binary }));
  const manifest = {
    version: 1,
    kind: 'ubuntu-baseline-executable',
    key,
    source: proof,
    environment,
    binary,
  };
  const temporary = options.cacheDirectory + `.${process.pid}.part`;
  fs.mkdirSync(temporary, { recursive: false, mode: 0o700 });
  try {
    fs.writeFileSync(path.join(temporary, 'binary.gz'), compressed, { flag: 'wx', mode: 0o600 });
    write(path.join(temporary, 'manifest.json'), manifest);
    fs.renameSync(temporary, options.cacheDirectory);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
  return {
    status: 'exported',
    key,
    binary_sha256: manifest.binary.sha256,
    directory: options.cacheDirectory,
  };
}

export async function validateBaselineCache(input, { restore = false } = {}) {
  const options = optionsPaths(input),
    inspect = input.inspect ?? inspectBaselineEnvironment;
  let temporary;
  try {
    const { manifest, bytes } = cachePayload(options.cacheDirectory);
    const proof = sourceProof(options);
    if (!isDeepStrictEqual(proof, manifest.source))
      throw new Error('Baseline source or overlay changed');
    temporary = fs.mkdtempSync(path.join(options.buildDirectory, '.baseline-cache-'));
    const candidate = path.join(temporary, executable);
    fs.writeFileSync(candidate, bytes, { flag: 'wx', mode: 0o700 });
    verifyProof(options, candidate, proof, manifest.binary.sha256, temporary);
    const environment = await inspect({ ...options, binary: candidate });
    if (!isDeepStrictEqual(environment, manifest.environment))
      throw new Error('Baseline compile/runtime dependency environment changed');
    if (restore) {
      if (fs.existsSync(options.binary)) regular(options.binary, maximumBinary);
      fs.renameSync(candidate, options.binary);
      recordBaselineBinary(options.evidenceFile, options.binary);
    }
    return {
      status: restore ? 'hit' : 'valid',
      key: manifest.key,
      binary_sha256: manifest.binary.sha256,
      fresh_runtime_validation_required: true,
    };
  } catch (error) {
    return {
      status: 'miss',
      reason: String(error),
      fallback: 'build baseline from its fresh verified source',
    };
  } finally {
    if (temporary) fs.rmSync(temporary, { recursive: true, force: true });
  }
}
export const restoreBaselineCache = (options) => validateBaselineCache(options, { restore: true });

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [action, cache, checkout, buildDirectory, evidenceFile, requestFile, reportFile, ...extra] =
    process.argv.slice(2);
  if (!['export', 'restore', 'validate'].includes(action) || !reportFile || extra.length)
    throw new Error(
      'Usage: baseline-cache.mjs {export|restore|validate} CACHE CHECKOUT BUILD EVIDENCE REQUEST REPORT',
    );
  const options = {
    cacheDirectory: cache,
    checkout,
    buildDirectory,
    evidenceFile,
    request: read(requestFile),
  };
  let report;
  try {
    report = await (action === 'export'
      ? exportBaselineCache(options)
      : action === 'restore'
        ? restoreBaselineCache(options)
        : validateBaselineCache(options));
  } catch (error) {
    if (action !== 'export') throw error;
    report = { status: 'unavailable', reason: String(error), baseline_build_unchanged: true };
  }
  write(reportFile, report);
  console.log(JSON.stringify(report));
}
