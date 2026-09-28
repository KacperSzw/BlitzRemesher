import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {writeFile} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.BLITZ_PLAYWRIGHT_MODULE || 'playwright');
const board = resolve(process.argv[2] || dirname(fileURLToPath(import.meta.url)));
const browser = await chromium.launch({headless: true,
  ...(process.env.BLITZ_CHROMIUM ? {executablePath: process.env.BLITZ_CHROMIUM} : {}),
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--renderer-process-limit=2']});
const errors = [], requests = [], checks = {desktop_width: 1440, mobile_width: 390, runs: []};
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}, deviceScaleFactor: 1});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  page.on('request', request => {if (/^https?:/.test(request.url())) requests.push(request.url());});
  await page.goto(pathToFileURL(join(board, 'index.html')).href);
  await page.waitForFunction(() => document.documentElement.dataset.ready === 'true');
  assert.equal(await page.locator('[role="tab"]').count(), 4);
  assert.match(await page.locator('#score-formula').textContent(),
    /average triangle count of LOD 1–7 divided by LOD 0/);
  assert.match(await page.locator('.headroom-note').textContent(),
    /does not prove that a lower-triangle proposal/);
  for (let index = 0; index < 4; ++index) {
    await page.locator(`#run-tab-${index}`).click();
    const cap = index < 2 ? 3 : 4, fallback = index % 2 === 1;
    assert.match(await page.locator(`#run-tab-${index}`).textContent(),
      new RegExp(`${cap} px · ${fallback ? 'E' : 'B'}`));
    assert.equal(await page.locator('.asset').count(), 8,
      `Run ${index} must show all eight pilot assets`);
    const note = await page.locator('#pilot-note').textContent();
    assert.match(note, /Pilot-only configured results/);
    assert.match(note, /triangle-first/);
    assert.match(note, /3 px and 4 px source-cap exports match/);
    assert.equal(await page.locator('#pilot-note a[href$="/summary.json"]').count(), 2);
    assert.match(await page.locator('#run-description').textContent(),
      new RegExp(`rises to the ${cap} px cap; each adjacent step allows 2 px`));
    const direct = await page.evaluate(() => {
      const {data, state} = window.areaBoard;
      return data.dense_checks[data.runs[state.run].id] ?? [];
    });
    assert.equal(await page.locator('#dense-note').isVisible(), direct.length > 0,
      'Finite rotated dense checks must appear on their direct run tab');
    if (direct.length) {
      const denseNote = await page.locator('#dense-note').textContent();
      for (const value of ['Finite rotated dense tail checks', '706-view',
        'tightest measured pixel margin', 'do not establish all-view robustness'])
        assert.ok(denseNote.includes(value), `Missing finite dense context: ${value}`);
      assert.equal(await page.locator('#dense-note a').count(), direct.length);
      for (const audit of direct)
        assert.equal(await page.locator(`#dense-note a[href="../${audit.file}"]`).count(), 1);
      await page.locator('#dense-note summary').click();
      assert.equal(await page.locator('#dense-note a').first().isVisible(), true,
        'Raw rotated audit links must be discoverable');
      await page.locator('#dense-note summary').click();
      if (fallback) {
        const firstSeed = direct.filter(audit => audit.seed === 3665436710);
        assert.equal(new Set(firstSeed.map(audit => audit.asset_id)).size, 8,
          'The first direct E rotated seed must cover all eight pilot assets');
        assert.match(denseNote, /first rotated seed covers 8\/8 assets with 8 passing audits/);
        assert.ok(direct.every(audit => audit.passed),
          'Every recorded E finite dense tail check must pass');
      }
      if (index === 1) {
        assert.ok(direct.filter(audit => audit.asset_name.includes('Bell')).length >= 2,
          'Bell second rotation is missing');
      }
    }
    const result = await page.evaluate(() => {
      const {data, state} = window.areaBoard, active = data.runs[state.run];
      const rows = [...document.querySelectorAll('.asset')];
      const painted = [...document.querySelectorAll('#assets canvas')].map(canvas => {
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) ++count;
        return count;
      });
      return {run: active.id, level: state.level, assets: rows.length, score: active.score,
        cap: active.source_cap, fallback: active.topology_fallback, extras: active.extra_proposals,
        same_cap_pair: active.identical_assets, other_cap_pair: active.other_cap_identical_assets,
        direct_dense_runs: (data.dense_checks[active.id] ?? []).length,
        inherited_dense_assets: active.inherited_dense_assets?.length ?? 0,
        finite_first_seed_assets: active.finite_first_seed_assets ?? 0,
        stable_order: rows.every((row, i) => row.querySelector('.asset-context').textContent.includes(data.pilot_ids[i])),
        selected_geometry: rows.every((row, i) => row.querySelector('canvas[data-level="7"]') && active.assets[i].levels[7].geometry),
        limits_valid: active.assets.every(asset => asset.levels.slice(1).every(level =>
          level.source.px_limit <= active.source_cap + 1e-12 &&
          Math.abs(level.adjacent.px_limit - 2) < 1e-12 &&
          level.source.area <= .5 + 1e-12 && level.adjacent.area <= .5 + 1e-12)),
        min_painted_pixels: Math.min(...painted),
        horizontal_overflow: document.documentElement.scrollWidth > innerWidth};
    });
    assert.equal(result.cap, cap);
    assert.equal(result.fallback, fallback);
    assert.equal(result.stable_order, true);
    assert.equal(result.selected_geometry, true);
    assert.equal(result.limits_valid, true);
    assert.ok(result.min_painted_pixels > 20, `Run ${index} has a blank geometry canvas`);
    assert.equal(result.horizontal_overflow, false);
    assert.equal(result.other_cap_pair, 8, 'The two source caps must show their recorded matching exports');
    if (fallback) {
      assert.ok(result.extras > 0);
      assert.match(await page.locator('#comparison').textContent(), /E versus B under the same/);
      assert.match(await page.locator('#comparison').textContent(), /extra topology proposals/);
    } else {
      assert.equal(result.extras, 0);
      assert.equal(result.inherited_dense_assets, result.same_cap_pair);
      assert.equal(result.finite_first_seed_assets, 8);
      assert.match(await page.locator('#dense-note').textContent(),
        /finite first-seed evidence covers 8\/8 B assets/);
    }
    const firstAsset = await page.evaluate(() => {
      const {data, state} = window.areaBoard, asset = data.runs[state.run].assets[0];
      return {expected: (100 * asset.chain_ratio).toFixed(1) + '%',
        visible: document.querySelector('.asset[data-asset="0"] .chain-retention strong').textContent,
        finalLabel: document.querySelector('.asset[data-asset="0"] .metric-lead small').textContent};
    });
    assert.equal(firstAsset.visible, firstAsset.expected);
    assert.equal(firstAsset.finalLabel, 'Final retention');
    checks.runs.push(result);
    await page.screenshot({path: join(board, `run-${index + 1}.png`), fullPage: true});
  }
  await page.locator('#run-tab-1').click();
  await page.locator('#lod').fill('4');
  assert.equal(await page.locator('#lod-output').textContent(), 'LOD 4');
  assert.equal(await page.locator('canvas[data-level="4"]').count(), 8);
  const before = await page.locator('canvas[data-asset="3"][data-level="4"]').evaluate(c => c.toDataURL());
  await page.locator('[data-mode="wire"]').click();
  const after = await page.locator('canvas[data-asset="3"][data-level="4"]').evaluate(c => c.toDataURL());
  assert.notEqual(after, before, 'Wireframe mode must change the rendered mesh');
  await page.locator('[data-mode="clay"]').click();
  await page.locator('#lod').fill('7');
  checks.lod_selection_checked = true;
  checks.wireframe_checked = true;
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false,
    'Mobile layout must fit');
  await page.screenshot({path: join(board, 'mobile.png'), fullPage: true});
  checks.mobile_no_horizontal_overflow = true;
  assert.deepEqual(errors, [], 'Browser errors');
  assert.deepEqual(requests, [], 'Board must work offline');
  checks.browser = await browser.version();
  checks.browser_errors = errors;
  checks.network_requests = requests;
  await writeFile(join(board, 'browser-checks.json'), JSON.stringify(checks, null, 2) + '\n');
  console.log(JSON.stringify(checks, null, 2));
} finally {
  await browser.close();
}
