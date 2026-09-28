// Render the recorded Round 4 meshes into a compact, looping README GIF.
// BLITZ_PLAYWRIGHT_MODULE may point to an existing Playwright installation.
// BLITZ_CHROMIUM and BLITZ_FFMPEG may select system binaries on NixOS.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createRequire} from 'node:module';
import {copyFile, mkdir, mkdtemp, readFile, rm, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.BLITZ_PLAYWRIGHT_MODULE || 'playwright');
const root = resolve(process.argv[2] || '.');
const output = resolve(process.argv[3] || join(root, 'docs/media'));
const board = join(root, 'research/round4/board');
const gif = join(output, 'lod-chain-showcase.gif');
const poster = join(output, 'lod-chain-showcase.png');
const ids = ['ph_moon_rock_02', 'ph_metal_stool_02', 'si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652'];
const fps = 10;
const framesPerLevel = 4;
const levelsPerAsset = 8;
const frames = ids.length * levelsPerAsset * framesPerLevel;
const maxBytes = 5 * 1024 * 1024;

const style = `
*{box-sizing:border-box}
html,body{margin:0;width:720px;height:405px;overflow:hidden;background:#000}
body{font-family:Arial,Helvetica,sans-serif;color:#fff}
#readme-demo{position:fixed;inset:0;width:720px;height:405px;background:#000}
#demo-name{position:absolute;top:20px;left:0;width:100%;text-align:center;font-size:22px;font-weight:600}
#demo-mesh{position:absolute;left:64px;top:58px;width:592px;height:272px}
#demo-mesh canvas{display:block;width:100%;height:100%;filter:brightness(1.65)}
#demo-label{position:absolute;bottom:22px;left:0;width:100%;text-align:center;
  font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
`;

function run(command, args) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command, args, {stdio: ['ignore', 'ignore', 'pipe']});
    let errors = '';
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', chunk => { errors += chunk; });
    child.on('error', reject);
    child.on('close', code => code === 0 ? resolveRun() : reject(new Error(`${command} exited ${code}: ${errors.slice(-2000)}`)));
  });
}

async function encode(ffmpeg, directory, palette, width, rate, colors) {
  const input = join(directory, 'frame-%03d.png');
  const filter = `fps=${rate},scale=${width}:-1:flags=lanczos`;
  await run(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-framerate', String(fps), '-i', input,
    '-vf', `${filter},palettegen=max_colors=${colors}:stats_mode=full`, palette]);
  await run(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-framerate', String(fps), '-i', input,
    '-i', palette, '-filter_complex', `[0:v]${filter}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle`,
    '-loop', '0', gif]);
  return (await stat(gif)).size;
}

await mkdir(output, {recursive: true});
const temporary = await mkdtemp(join(tmpdir(), 'blitz-readme-'));
const failures = [];
const requests = [];
const browser = await chromium.launch({
  headless: true,
  ...(process.env.BLITZ_CHROMIUM ? {executablePath: process.env.BLITZ_CHROMIUM} : {}),
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--renderer-process-limit=2', '--num-raster-threads=2']
});
try {
  const manifest = JSON.parse(await readFile(join(board, 'manifest.json'), 'utf8'));
  assert.equal(manifest.run, 'r4-final-s512-cap8');
  const page = await browser.newPage({viewport: {width: 720, height: 405}, deviceScaleFactor: 1});
  page.on('pageerror', error => failures.push(error.message));
  page.on('console', message => { if (message.type() === 'error') failures.push(message.text()); });
  page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
  await page.goto(pathToFileURL(join(board, 'index.html')).href);
  await page.waitForFunction(() => document.documentElement.dataset.ready === 'true');
  const summary = await page.evaluate(expectedIds => {
    const {data, renderer} = window.blitzBoard;
    return {
      run: data.run,
      settings: data.settings,
      assets: expectedIds.map(id => {
        const index = data.assets.findIndex(asset => asset.id === id);
        if (index < 0) throw new Error(`Missing board asset ${id}`);
        const asset = data.assets[index];
        return {id, index, name: asset.name, sourceUrl: asset.source_url, triangles: asset.lods.map(lod => lod.triangles),
          screens: asset.lods.map(lod => lod.screen_pixels), runtime: asset.runtime_levels.length,
          accepted: asset.lods.every((lod, level) => level === 0 ||
            (lod.adjacent.passed && lod.adjacent.complete && !lod.adjacent.resource_limited &&
             lod.source.passed && lod.source.complete && !lod.source.resource_limited)),
          geometryMatches: asset.lods.every((lod, level) => renderer.meshes[index][level].count / 3 === lod.triangles)};
      })
    };
  }, ids);
  assert.equal(summary.run, manifest.run);
  assert.deepEqual(summary.settings.audit_views, {orthographic: 12, perspective: 4, seed: 2971082790});
  assert.equal(summary.settings.chain, 'hybrid');
  assert.equal(summary.settings.output, 'rebuild');
  assert.equal(summary.settings.profile, 'coverage');
  assert.equal(summary.settings.base_pixels, 512);
  assert.equal(summary.settings.last_pixels, 16);
  assert.equal(summary.settings.levels, 8);
  for (const asset of summary.assets) {
    const record = manifest.assets.find(item => item.id === asset.id);
    assert.ok(record, `Missing provenance for ${asset.id}`);
    assert.equal(record.license, 'CC0-1.0');
    assert.equal(asset.sourceUrl, record.source_url);
    assert.deepEqual(asset.triangles, record.triangles);
    assert.equal(asset.triangles.length, 8);
    assert.equal(asset.screens[0], 512);
    assert.equal(asset.screens.at(-1), 16);
    assert.ok(asset.accepted && asset.geometryMatches, `Invalid saved chain for ${asset.id}`);
  }

  await page.addStyleTag({content: style});
  await page.evaluate(assets => {
    document.querySelector('.sheet').style.display = 'none';
    const demo = document.createElement('div');
    demo.id = 'readme-demo';
    demo.innerHTML = '<div id="demo-name"></div><div id="demo-mesh"><canvas id="demo-canvas"></canvas></div><div id="demo-label"></div>';
    document.body.append(demo);
    window.readmeDemo = {assets};
  }, summary.assets);

  for (let frame = 0; frame < frames; ++frame) {
    await page.evaluate(({frame, framesPerLevel, levelsPerAsset}) => {
      const {data, state, renderer} = window.blitzBoard;
      const assetIndex = Math.floor(frame / (levelsPerAsset * framesPerLevel));
      const level = Math.floor((frame % (levelsPerAsset * framesPerLevel)) / framesPerLevel);
      const item = window.readmeDemo.assets[assetIndex];
      const asset = data.assets[item.index];
      const lod = asset.lods[level];
      state.mode = 'clay';
      state.scale = 'enlarged';
      state.cameras[item.index] = {yaw: asset.yaw, pitch: asset.pitch};
      document.getElementById('demo-name').textContent = item.name;
      document.getElementById('demo-label').textContent =
        'LOD ' + level + ' · ' + Math.round(lod.screen_pixels) + ' px · ' + lod.triangles.toLocaleString('en-US') + ' triangles';
      renderer.draw(document.getElementById('demo-canvas'), item.index, level);
    }, {frame, framesPerLevel, levelsPerAsset});
    const framePath = join(temporary, `frame-${String(frame).padStart(3, '0')}.png`);
    await page.screenshot({path: framePath, animations: 'disabled'});
    if (frame === 0) await copyFile(framePath, poster);
  }
  assert.deepEqual(requests, [], 'Demo must not make network requests');
  assert.deepEqual(failures, [], 'Demo must not log browser errors');
  const ffmpeg = process.env.BLITZ_FFMPEG || 'ffmpeg';
  const palette = join(temporary, 'palette.png');
  let bytes;
  let mode;
  for (const candidate of [{width: 720, rate: 10, colors: 128}, {width: 640, rate: 10, colors: 112}, {width: 640, rate: 8, colors: 96}]) {
    bytes = await encode(ffmpeg, temporary, palette, candidate.width, candidate.rate, candidate.colors);
    mode = candidate;
    if (bytes <= maxBytes) break;
  }
  assert.ok(bytes <= maxBytes, `GIF exceeds 5 MiB after fallback encoding (${bytes} bytes)`);
  console.log(JSON.stringify({gif, poster, bytes, frames, encoded: mode, assets: summary.assets.map(a => ({name: a.name, source: a.triangles[0], final: a.triangles.at(-1), runtime: a.runtime})), networkRequests: requests.length, browserErrors: failures.length}, null, 2));
} finally {
  await browser.close();
  await rm(temporary, {recursive: true, force: true});
}
