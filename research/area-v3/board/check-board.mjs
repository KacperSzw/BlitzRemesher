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
  assert.equal(await page.locator('[role="tab"]').count(), 5);
  assert.match(await page.locator('#score-formula').textContent(), /average triangle count of LOD 1–7 divided by LOD 0/);
  assert.match(await page.locator('.headroom-note').textContent(), /does not prove that a lower-triangle proposal/);
  for (let run = 0; run < 5; ++run) {
    await page.locator(`#run-tab-${run}`).click();
    assert.equal(await page.locator('.asset').count(), 8, `Run ${run} must show all eight assets`);
    assert.equal(await page.locator('#dense-check').isVisible(), run === 1,
      'Only the cap-only run has this independent dense failure');
    assert.equal(await page.locator('#topology-check').isVisible(), run === 3,
      'Only the experimental topology run has its independent dense summary');
    assert.equal(await page.locator('#fallback-check').isVisible(), run === 4,
      'Only the post-hoc conditional fallback run has its extra-work summary');
    if (run === 1) {
      const finding = await page.locator('#dense-check').textContent();
      assert.match(finding, /51\.29%/);
      assert.match(finding, /42\.73%/);
      assert.match(finding, /not ready for promotion/);
      assert.equal(await page.locator('#dense-check a[href="../dense-tree-cap-0.5.json"]').count(), 1);
      assert.match(await page.locator('.asset[data-asset="3"] .status').textContent(), /dense LOD 7 failed/);
    }
    if (run === 3) {
      const finding = await page.locator('#topology-check').textContent();
      for (const value of ['844→48', '1,273→64', '51.41%', '50.71%', '50.98%', 'regresses', 'not a promoted preset'])
        assert.ok(finding.includes(value), `Missing experimental finding: ${value}`);
      for (const name of ['tree', 'rock', 'hatch', 'bell'])
        assert.equal(await page.locator(`#topology-check a[href="../dense-${name}-topology-relaxed.json"]`).count(), 1);
      assert.match(await page.locator('.asset[data-asset="3"] .status').textContent(), /dense LOD 7 failed/);
      assert.match(await page.locator('.asset[data-asset="5"] .status').textContent(), /dense LOD 7 failed/);
      assert.match(await page.locator('.asset[data-asset="6"] .status').textContent(), /dense tail passed/);
      assert.match(await page.locator('.asset[data-asset="7"] .status').textContent(), /dense tail passed/);
      assert.match(await page.locator('#comparison').textContent(), /global eight-asset SCORE regresses/);
    }
    if (run === 4) {
      const finding = await page.locator('#fallback-check').textContent();
      for (const value of ['1,273→128', '844→96', '19.05%', '34.54%', '51.29%',
        'same corrected binary', '5 extra proposals', 'no global preset promotion'])
        assert.ok(finding.includes(value), `Missing conditional fallback finding: ${value}`);
      for (const name of ['bell-topology-fallback-v2', 'hatch-topology-fallback-v2', 'tree-topology-fallback-v2'])
        assert.equal(await page.locator(`#fallback-check a[href="../dense-${name}.json"]`).count(), 1);
      assert.equal(await page.locator('#fallback-check a[href="../../runs/area-v3-cap-0.5-fallback-v2-b/summary.json"]').count(), 1);
      assert.match(await page.locator('.asset[data-asset="3"] .status').textContent(), /Unchanged B tree · rotated dense failed/);
      assert.match(await page.locator('.asset[data-asset="6"] .status').textContent(), /dense tail passed · extra proposal/);
      assert.match(await page.locator('.asset[data-asset="7"] .status').textContent(), /dense tail passed · extra proposal/);
      assert.match(await page.locator('#comparison').textContent(), /0\.0556 points/);
      assert.match(await page.locator('#comparison').textContent(), /same-binary v2 B control/);
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
      return {run: active.id, level: state.level, assets: rows.length,
        stable_order: rows.every((row, i) => row.querySelector('.asset-context').textContent.includes(data.pilot_ids[i])),
        selected_geometry: rows.every((row, i) => row.querySelector('canvas[data-level="7"]') && active.assets[i].levels[7].geometry),
        min_painted_pixels: Math.min(...painted), horizontal_overflow: document.documentElement.scrollWidth > innerWidth};
    });
    assert.equal(result.stable_order, true);
    assert.equal(result.selected_geometry, true);
    assert.ok(result.min_painted_pixels > 20, `Run ${run} has a blank geometry canvas`);
    assert.equal(result.horizontal_overflow, false);
    const firstAsset = await page.evaluate(() => {
      const {data, state} = window.areaBoard, asset = data.runs[state.run].assets[0];
      return {expected: (100 * asset.chain_ratio).toFixed(1) + '%',
        visible: document.querySelector('.asset[data-asset="0"] .chain-retention strong').textContent,
        finalLabel: document.querySelector('.asset[data-asset="0"] .metric-lead small').textContent};
    });
    assert.equal(firstAsset.visible, firstAsset.expected);
    assert.equal(firstAsset.finalLabel, 'Final retention');
    checks.runs.push(result);
    await page.screenshot({path: join(board, `run-${run + 1}.png`), fullPage: true});
  }
  const shelves = await page.evaluate(() => {
    const [, fixed, adaptive] = window.areaBoard.data.runs;
    return {fixedChain: fixed.assets[0].chain_ratio, adaptiveChain: adaptive.assets[0].chain_ratio,
      fixedFinal: fixed.assets[0].levels[7].triangles, adaptiveFinal: adaptive.assets[0].levels[7].triangles};
  });
  assert.ok(shelves.adaptiveFinal > shelves.fixedFinal && shelves.adaptiveChain < shelves.fixedChain,
    'The shelves demonstrate why final retention alone does not explain SCORE');
  checks.score_formula_and_chain_retention_checked = true;
  checks.experimental_topology_checked = true;
  checks.conditional_fallback_checked = true;
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
  checks.rotated_dense_failure_visible = true;
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Mobile layout must fit');
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
