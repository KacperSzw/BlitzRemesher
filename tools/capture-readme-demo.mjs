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
const fps = 10;
const frames = 90;
const maxBytes = 5 * 1024 * 1024;
const ids = ['ph_moon_rock_02', 'ph_metal_stool_02', 'si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652'];

const style = `
*{box-sizing:border-box}html,body{margin:0;width:720px;height:405px;overflow:hidden}
body{font-family:Arial,Helvetica,sans-serif;color:#edf3f5;background:#0b1119}
#readme-demo{width:720px;height:405px;overflow:hidden;position:fixed;inset:0;padding:20px 25px;
  background:radial-gradient(circle at 78% 20%,#19313c 0%,transparent 45%),
  linear-gradient(145deg,#111d29 0%,#0b1119 70%)}
#readme-demo:before{content:"";position:absolute;inset:0;pointer-events:none;opacity:.12;
  background-image:linear-gradient(#8bb6bb 1px,transparent 1px),linear-gradient(90deg,#8bb6bb 1px,transparent 1px);
  background-size:38px 38px;mask-image:linear-gradient(to bottom,#000,transparent 92%)}
#readme-demo>*{position:relative}#readme-demo canvas{display:block;width:100%;height:100%;filter:brightness(1.75) saturate(1.2)}
.demo-top{height:28px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #344c58;padding-bottom:11px}
.demo-brand{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:800;letter-spacing:.15em}
.demo-mark{width:17px;height:17px;display:grid;place-items:center;background:#ef663e;color:#fff;font-size:14px;line-height:1}
.demo-top-right{font:10px Arial,Helvetica,sans-serif;letter-spacing:.13em;color:#95abb4}
.demo-eyebrow{font-size:10px;letter-spacing:.17em;font-weight:700;color:#78c7c5}
.demo-poster{padding-top:24px}.demo-poster h1{font-size:31px;line-height:1.07;letter-spacing:-.04em;margin:9px 0 17px}
.demo-poster h1 span{color:#ff8259}.demo-poster-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.demo-poster-card{height:196px;border:1px solid #36505b;background:linear-gradient(150deg,#1b2c39,#101b26);border-radius:8px;padding:11px;overflow:hidden}
.demo-poster-name{font-size:13px;font-weight:700}.demo-poster-type{font-size:9px;letter-spacing:.12em;color:#8da5af;margin-top:2px}
.demo-poster-meshes{height:119px;display:grid;grid-template-columns:1fr 1fr;gap:2px;align-items:stretch}
.demo-poster-meshes>div{min-width:0;height:119px}.demo-poster-count{font-size:14px;font-weight:800;letter-spacing:-.02em;white-space:nowrap}
.demo-poster-count span{font-size:11px;color:#ff8259;margin:0 2px}.demo-poster-note{font-size:9px;color:#aabdc3;margin-top:3px}
.demo-scene{padding-top:17px}.demo-scene-heading{display:flex;align-items:end;justify-content:space-between;gap:12px;height:48px}
.demo-scene-title{font-size:27px;line-height:1;letter-spacing:-.04em;font-weight:800;margin-top:4px}
.demo-reduction{color:#6fd7c4;font-size:26px;font-weight:800;letter-spacing:-.04em}
.demo-reduction small{display:block;color:#a6c4c7;font-size:9px;letter-spacing:.12em;text-align:right;margin-top:1px}
.demo-views{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.demo-view{height:211px;padding:10px 13px 8px;border:1px solid #38525e;border-radius:9px;background:linear-gradient(150deg,#20313e,#14212d);position:relative;overflow:hidden}
.demo-view.selected{border-color:#457e80;background:linear-gradient(150deg,#1d3b43,#14262e)}
.demo-view:after{content:"";position:absolute;bottom:42px;left:12%;right:12%;height:1px;background:linear-gradient(90deg,transparent,#91b5b370,transparent)}
.demo-view-head{display:flex;justify-content:space-between;align-items:center;color:#a6bdc5;font-size:10px;font-weight:700;letter-spacing:.1em}
.demo-view-head strong{color:#77ddcb;font-size:10px}
.demo-mesh{height:149px}.demo-view-foot{display:flex;align-items:baseline;justify-content:space-between;position:relative;z-index:1}
.demo-view-count{font-size:21px;font-weight:800;letter-spacing:-.035em}.demo-view-count span{font-size:10px;font-weight:500;color:#a7bbc3}
.demo-view-status{font-size:9px;font-weight:700;letter-spacing:.04em;color:#77ddcb}
.demo-track{margin-top:13px;display:grid;grid-template-columns:repeat(8,1fr);gap:5px}
.demo-step{height:29px;border-top:3px solid #36505a;padding-top:5px;color:#78949f;font-size:9px;font-weight:700;letter-spacing:.04em}
.demo-step.active{border-color:#ff8259;color:#fff}.demo-step.done{border-color:#5ca99f;color:#afc4c9}
.demo-step span{display:block;font-size:8px;margin-top:2px;color:#829ca5}
.demo-foot{position:absolute!important;left:25px;right:25px;bottom:14px;display:flex;justify-content:space-between;align-items:center;
  font-size:9px;letter-spacing:.09em;color:#a1b7bf}.demo-foot strong{color:#78d5c3}
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
  const page = await browser.newPage({viewport: {width: 720, height: 405}, deviceScaleFactor: 1, reducedMotion: 'reduce'});
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
    const fmt = value => value.toLocaleString('en-US');
    const percent = asset => (100 * (1 - asset.triangles.at(-1) / asset.triangles[0])).toFixed(1);
    document.querySelector('.sheet').style.display = 'none';
    const demo = document.createElement('div');
    demo.id = 'readme-demo';
    demo.innerHTML = `
      <header class="demo-top"><div class="demo-brand"><span class="demo-mark">ϟ</span>BLITZREMESHER</div><div class="demo-top-right">C++20 · STATIC MESH LOD</div></header>
      <section class="demo-poster" id="demo-poster">
        <div class="demo-eyebrow">REAL OUTPUT GEOMETRY · 512 → 16 PX</div>
        <h1>Fewer triangles.<br><span>Measured transitions.</span></h1>
        <div class="demo-poster-grid">${assets.map((a, i) => `<article class="demo-poster-card">
          <div class="demo-poster-name">${a.name}</div><div class="demo-poster-type">${['ROCK','MANUFACTURED','COMPLEX SCAN'][i]}</div>
          <div class="demo-poster-meshes"><div><canvas class="demo-poster-source" data-case="${i}"></canvas></div><div><canvas class="demo-poster-final" data-case="${i}"></canvas></div></div>
          <div class="demo-poster-count">${fmt(a.triangles[0])}<span>→</span>${fmt(a.triangles.at(-1))}</div>
          <div class="demo-poster-note">triangles · −${percent(a)}%</div></article>`).join('')}</div>
      </section>
      <section class="demo-scene" id="demo-scene" hidden>
        <div class="demo-scene-heading"><div><div class="demo-eyebrow" id="demo-category"></div><div class="demo-scene-title" id="demo-name"></div></div>
          <div class="demo-reduction" id="demo-reduction"></div></div>
        <div class="demo-views">
          <div class="demo-view"><div class="demo-view-head"><span>LOD 0 · SOURCE</span><span>512 PX</span></div>
            <div class="demo-mesh"><canvas id="demo-source"></canvas></div>
            <div class="demo-view-foot"><div class="demo-view-count" id="demo-source-count"></div><div class="demo-view-status">UNCHANGED</div></div></div>
          <div class="demo-view selected"><div class="demo-view-head"><span id="demo-lod"></span><strong id="demo-screen"></strong></div>
            <div class="demo-mesh"><canvas id="demo-selected"></canvas></div>
            <div class="demo-view-foot"><div class="demo-view-count" id="demo-selected-count"></div><div class="demo-view-status" id="demo-audit"></div></div></div>
        </div><div class="demo-track" id="demo-track"></div>
      </section>
      <footer class="demo-foot"><span>HYBRID REBUILD · COVERAGE PROFILE</span><strong>16 CONFIGURED AUDIT CAMERAS</strong></footer>`;
    document.body.append(demo);
    window.readmeDemo = {assets, fmt};
  }, summary.assets);

  for (let frame = 0; frame < frames; ++frame) {
    await page.evaluate(({frame, total}) => {
      const demo = window.readmeDemo;
      const {data, state, renderer} = window.blitzBoard;
      const poster = document.getElementById('demo-poster');
      const scene = document.getElementById('demo-scene');
      const isPoster = frame < 7 || frame >= total - 5;
      poster.hidden = !isPoster;
      scene.hidden = isPoster;
      state.mode = 'clay';
      state.scale = 'enlarged';
      if (isPoster) {
        demo.assets.forEach((item, i) => {
          const asset = data.assets[item.index];
          state.cameras[item.index] = {yaw: asset.yaw, pitch: asset.pitch};
          renderer.draw(document.querySelector(`.demo-poster-source[data-case="${i}"]`), item.index, 0);
          renderer.draw(document.querySelector(`.demo-poster-final[data-case="${i}"]`), item.index, 7);
        });
        return;
      }
      const relative = frame - 7;
      const caseIndex = Math.min(2, Math.floor(relative / 26));
      const progress = (relative % 26) / 25;
      const item = demo.assets[caseIndex];
      const asset = data.assets[item.index];
      const level = Math.min(7, 1 + Math.floor(progress * 7));
      const lod = asset.lods[level];
      state.cameras[item.index] = {yaw: asset.yaw + 0.52 * progress, pitch: asset.pitch + 0.07 * Math.sin(progress * Math.PI)};
      state.mode = progress >= 0.44 && progress <= 0.6 ? 'wire' : 'clay';
      const set = (id, value) => { document.getElementById(id).textContent = value; };
      set('demo-category', `${String(caseIndex + 1).padStart(2, '0')} / ${['ROCK','MANUFACTURED','COMPLEX SCAN'][caseIndex]}`);
      set('demo-name', item.name);
      document.getElementById('demo-reduction').innerHTML = `−${(100 * (1 - asset.lods.at(-1).triangles / asset.lods[0].triangles)).toFixed(1)}%<small>${item.runtime} RUNTIME / 8 AUDIT SLOTS</small>`;
      set('demo-source-count', demo.fmt(asset.lods[0].triangles));
      set('demo-lod', `LOD ${level} · ${state.mode === 'wire' ? 'WIREFRAME' : 'REDUCED'}`);
      set('demo-screen', `${Math.round(lod.screen_pixels)} PX`);
      set('demo-selected-count', demo.fmt(lod.triangles));
      set('demo-audit', '✓ AUDIT PASSED');
      document.getElementById('demo-track').innerHTML = asset.lods.map((entry, index) =>
        `<div class="demo-step ${index === level ? 'active' : index < level ? 'done' : ''}">LOD ${index}<span>${index === 0 ? 'SOURCE' : `${Math.round(entry.screen_pixels)} PX`}</span></div>`).join('');
      renderer.draw(document.getElementById('demo-source'), item.index, 0);
      renderer.draw(document.getElementById('demo-selected'), item.index, level);
    }, {frame, total: frames});
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
