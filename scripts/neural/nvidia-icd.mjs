// Preserve the host driver's NVIDIA entry; choosing another backend is never a fallback.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const digest = (value) => createHash('sha256').update(value).digest('hex');

export function selectNvidiaIcd(contents, { sourcePath, selectedPath, requested = 'vendor' }) {
  if (
    typeof contents !== 'string' ||
    !path.isAbsolute(sourcePath ?? '') ||
    !path.isAbsolute(selectedPath ?? '') ||
    !['vendor', 'glx', 'egl'].includes(requested)
  )
    throw new Error('Invalid NVIDIA ICD selection');
  const manifest = JSON.parse(contents),
    library = manifest?.ICD?.library_path;
  if (
    typeof library !== 'string' ||
    !library ||
    /[\x00-\x1f]/.test(library) ||
    !/^\d+\.\d+\.\d+$/.test(manifest.file_format_version ?? '') ||
    !/^\d+\.\d+\.\d+$/.test(manifest.ICD.api_version ?? '')
  )
    throw new Error('Invalid NVIDIA ICD manifest');
  const backend = new Map([
    ['libGLX_nvidia.so.0', 'glx'],
    ['libEGL_nvidia.so.0', 'egl'],
  ]).get(path.basename(library));
  if (!backend) throw new Error('Unsupported NVIDIA Vulkan ICD entry');
  if (requested !== 'vendor' && requested !== backend)
    throw new Error(`Requested NVIDIA ${requested} ICD but vendor manifest selects ${backend}`);
  // A soname uses the system library search path. A relative directory is
  // relative to the original manifest, which changes when copying into results.
  const selectedLibrary =
    !path.isAbsolute(library) && library.includes('/')
      ? path.resolve(path.dirname(sourcePath), library)
      : library;
  let selected = contents;
  if (selectedLibrary !== library) {
    manifest.ICD.library_path = selectedLibrary;
    selected = JSON.stringify(manifest, null, 2) + '\n';
  }
  return {
    contents: selected,
    provenance: {
      version: 1,
      requested,
      backend,
      source_path: sourcePath,
      source_sha256: digest(contents),
      source_library_path: library,
      selected_path: selectedPath,
      selected_sha256: digest(selected),
      selected_library_path: selectedLibrary,
      file_format_version: manifest.file_format_version,
      api_version: manifest.ICD.api_version,
    },
  };
}

export function verifyNvidiaIcdSelection(file, requested) {
  const provenance = JSON.parse(fs.readFileSync(file, 'utf8'));
  const original = fs.readFileSync(provenance.original_copy_path, 'utf8');
  const expected = selectNvidiaIcd(original, {
    sourcePath: provenance.source_path,
    selectedPath: provenance.selected_path,
    requested,
  });
  for (const [key, value] of Object.entries(expected.provenance))
    if (provenance[key] !== value) throw new Error('NVIDIA ICD provenance mismatch: ' + key);
  if (digest(fs.readFileSync(provenance.selected_path)) !== provenance.selected_sha256)
    throw new Error('Selected NVIDIA ICD checksum mismatch');
  if (
    provenance.dependency_library_resolution !==
      (path.isAbsolute(provenance.selected_library_path) ? 'manifest_path' : 'ldconfig') ||
    !path.isAbsolute(provenance.dependency_library_path ?? '') ||
    digest(fs.readFileSync(provenance.dependency_library_path)) !==
      provenance.dependency_library_sha256
  )
    throw new Error('Checked NVIDIA ICD library checksum mismatch');
  return provenance;
}
