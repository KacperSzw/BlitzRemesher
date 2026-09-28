// Optional presentation export. The board itself has no JavaScript dependencies.
// BLITZ_PLAYWRIGHT_MODULE may point to an existing Playwright installation.
// BLITZ_CHROMIUM may select a system Chromium (useful on NixOS).
import {createRequire} from 'node:module';
import {mkdir, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.BLITZ_PLAYWRIGHT_MODULE || 'playwright');
const root = resolve(process.argv[2] || '.');
const output = resolve(process.argv[3] || root + '/research/board');
await mkdir(output, {recursive:true});
const browser = await chromium.launch({
  headless:true,
  ...(process.env.BLITZ_CHROMIUM ? {executablePath:process.env.BLITZ_CHROMIUM} : {}),
  args:['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--renderer-process-limit=2', '--num-raster-threads=2']
});
const failures = [];
try {
  const page = await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1});
  page.on('pageerror',error=>failures.push(error.message));
  page.on('console',message=>{if(message.type()==='error')failures.push(message.text());});
  // This must remain a portable, offline document.
  const requests = [];
  page.on('request',request=>{if(/^https?:/.test(request.url()))requests.push(request.url());});
  await page.goto(pathToFileURL(root + '/research/board/index.html').href);
  await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  const counts = await page.evaluate(()=>{
    const {data,renderer}=window.blitzBoard;
    return {
      cards:document.querySelectorAll('.lod-card').length,
      assets:data.assets.length,
      correctGeometry:renderer.meshes.every((levels,a)=>levels.every((mesh,l)=>mesh.count/3===data.assets[a].lods[l].triangles)),
      painted:[...document.querySelectorAll('.chain canvas')].map(canvas=>{
        const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
        let count=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])++count;return count;
      }),
      overflow:document.documentElement.scrollWidth>window.innerWidth,
      height:document.documentElement.scrollHeight
    };
  });
  assert.equal(counts.assets,4);
  assert.equal(counts.cards,32);
  assert.ok(counts.correctGeometry,'Displayed triangle counts must match the loaded geometry');
  assert.ok(counts.painted.every(n=>n>20),'Each LOD tile must contain a visible mesh');
  assert.equal(counts.overflow,false,'Desktop layout must fit the viewport');
  await page.screenshot({path:output+'/board.png',fullPage:true});
  await page.emulateMedia({media:'screen'});
  await page.pdf({path:output+'/board.pdf',width:'1600px',height:counts.height+'px',printBackground:true,margin:{top:0,bottom:0,left:0,right:0}});

  const initial = await page.locator('.chain canvas').first().evaluate(c=>c.toDataURL());
  await page.locator('.controls [data-mode="wire"]').click();
  const wire = await page.locator('.chain canvas').first().evaluate(c=>c.toDataURL());
  assert.notEqual(wire,initial,'Wireframe must change the rendered geometry');
  await page.screenshot({path:output+'/board-wireframe.png',fullPage:true});
  await page.locator('.controls [data-mode="silhouette"]').click();
  const silhouette = await page.locator('.chain canvas').first().evaluate(c=>c.toDataURL());
  assert.notEqual(silhouette,wire);
  await page.locator('.controls [data-mode="clay"]').click();

  await page.locator('.controls [data-scale="target"]').click();
  const targetPixels = await page.locator('.chain canvas').first().evaluate(c=>{
    const p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;
    for(let i=3;i<p.length;i+=4)if(p[i])++n;return n;
  });
  assert.ok(targetPixels>0&&targetPixels<counts.painted[0]/2,'Target scale should render a smaller, nonempty mesh');
  await page.locator('.controls [data-scale="enlarged"]').click();

  await page.locator('.lod-card[data-asset="2"][data-lod="7"]').click();
  assert.equal(await page.locator('#inspector').evaluate(d=>d.open),true);
  assert.equal(await page.locator('#selected-triangles').textContent(),'126 triangles');
  await page.locator('[data-level="0"]').click();
  assert.equal(await page.locator('#selected-triangles').textContent(),'6,532 triangles');
  const sameSource=await page.evaluate(()=>document.getElementById('source-view').toDataURL()===document.getElementById('selected-view').toDataURL());
  assert.ok(sameSource,'Comparing LOD 0 against itself must produce identical images');
  await page.locator('[data-level="7"]').click();
  const beforeRotate=await page.locator('#selected-view').evaluate(c=>c.toDataURL());
  await page.locator('#rotate-right').click();
  await page.waitForFunction(before=>document.getElementById('selected-view').toDataURL()!==before,beforeRotate);
  await page.locator('#inspect-reset').click();
  await page.locator('#inspector [data-mode="wire"]').click();
  await page.locator('#inspector').screenshot({path:output+'/stool-comparison.png'});
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#inspector').evaluate(d=>d.open),false);
  await page.locator('.controls [data-mode="clay"]').click();

  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(200);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth),false,'Mobile layout must not overflow horizontally');
  assert.equal(await page.locator('.lod-card').count(),32);
  await page.screenshot({path:output+'/mobile-preview.png',fullPage:true});
  assert.deepEqual(requests,[],'The board must not fetch external scripts, models or fonts');
  assert.deepEqual(failures,[],'The board must have no browser errors');
  const record={
    browser:await browser.version(), desktop:{width:1600,height:counts.height},
    mobile_width:390, examples:4, lod_tiles:32, all_tiles_painted:true,
    triangle_counts_match:true, modes_checked:['clay','wire','silhouette'],
    target_scale_checked:true, inspector_lod0_identity_checked:true,
    inspector_selection_and_rotation_checked:true, no_horizontal_overflow:true,
    network_requests:requests.length, browser_errors:failures
  };
  await writeFile(output+'/browser-checks.json',JSON.stringify(record,null,2)+'\n');
  console.log(JSON.stringify(record,null,2));
} finally {
  await browser.close();
}
