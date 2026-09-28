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
const input = resolve(process.argv[4] || root + '/research/board/index.html');
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
  await page.goto(pathToFileURL(input).href);
  await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  const counts = await page.evaluate(()=>{
    const {data,renderer}=window.blitzBoard;
    return {
      cards:document.querySelectorAll('.lod-card').length,
      assets:data.assets.length,
      foliageChains:data.assets.filter(a=>a.audit_scope==='card_geometry_only').length,
      expectedFoliageChains:data.foliage_chains||0,
      catalog:data.foliage_source_count>0,
      expectedCards:data.assets.reduce((n,a)=>n+a.lods.length,0),
      lastLevel:data.assets[0].lods.length-1,
      expectedSelected:data.assets[2].lods.at(-1).triangles,
      expectedSource:data.assets[2].lods[0].triangles,
      correctGeometry:renderer.meshes.every((levels,a)=>levels.every((mesh,l)=>mesh.count/3===data.assets[a].lods[l].triangles)),
      painted:[...document.querySelectorAll('.chain canvas')].map(canvas=>{
        const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
        let count=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])++count;return count;
      }),
      overflow:document.documentElement.scrollWidth>window.innerWidth,
      height:document.documentElement.scrollHeight
    };
  });
  assert.equal(counts.assets,4+counts.expectedFoliageChains);
  assert.equal(counts.foliageChains,counts.expectedFoliageChains);
  assert.equal(counts.cards,counts.expectedCards);
  assert.ok(counts.correctGeometry,'Displayed triangle counts must match the loaded geometry');
  assert.ok(counts.painted.every(n=>n>20),'Each LOD tile must contain a visible mesh');
  assert.equal(counts.overflow,false,'Desktop layout must fit the viewport');
  const chainMetadata=await page.evaluate(()=>window.blitzBoard.data.assets.every((a,i)=>{
    const row=document.querySelector('.asset[data-asset="'+i+'"]');
    const shared=a.lods.slice(1).filter(l=>l.shared_vertices).length;
    const storage=a.output_mode==='reuse'?'shared original buffer':!shared?'rebuilt / new buffers':shared===a.lods.length-1?'shared original buffer (rebuild mode)':'mixed source + new buffers (rebuild mode)';
    return row.querySelector('.chain-metadata').textContent.includes(Number(a.bake_seconds).toFixed(2)+' s')&&
      row.querySelector('.chain-metadata').textContent.includes(storage)&&
      [...row.querySelectorAll('.vertex-storage')].every((el,l)=>el.textContent===(a.lods[l].shared_vertices?'SOURCE VERTICES':'NEW VERTICES'));
  }));
  assert.ok(chainMetadata,'Every chain needs measured seconds and accurate vertex storage labels');
  assert.equal(await page.locator('.foliage-card').count(),0,'Nature examples belong in LOD rows, not a separate source gallery');
  if(counts.foliageChains){
    const correct=await page.evaluate(()=>window.blitzBoard.data.assets.filter(a=>a.audit_scope==='card_geometry_only').every(a=>a.lods.length>1&&a.bake_seconds>0&&a.bake_timing_scope==='generation_and_audit'));
    assert.ok(correct,'Nature rows must contain measured baked chains');
    const vegetation=await page.evaluate(()=>Boolean(window.blitzBoard.data.vegetation));
    if(vegetation){
      const checked=await page.evaluate(()=>{
        const d=window.blitzBoard.data;
        return document.querySelectorAll('.comparison-table tbody tr').length===d.vegetation.comparisons.length&&
          d.assets.every((a,i)=>a.dense_passed===undefined||document.querySelector('.asset[data-asset="'+i+'"] .dense-status').textContent.includes(a.dense_passed?'passed':'failed'));
      });
      assert.ok(checked,'Matched comparisons and independent dense failures must remain visible');
      for(const [index,name] of [[5,'fern-shared-candidate'],[7,'fern-rebuilt-candidate'],[9,'tree-shared-candidate'],[11,'tree-rebuilt-candidate']])
        await page.locator('.asset').nth(index).screenshot({path:output+'/'+name+'.png'});
      await page.locator('.comparison-scroll').screenshot({path:output+'/matched-comparisons.png'});
    }else{
      await page.locator('.asset').nth(4).screenshot({path:output+'/grass-chain.png'});
      await page.locator('.asset').nth(5).screenshot({path:output+'/fern-reuse-chain.png'});
    }
  }
  await page.locator('.asset').first().screenshot({path:output+'/chain-metadata.png'});
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

  const targetCanvas=page.locator('.chain canvas[data-asset="0"][data-lod="'+counts.lastLevel+'"]');
  await page.locator('.controls [data-scale="target"]').click();
  const targetPixels = await targetCanvas.evaluate(c=>{
    const p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;
    for(let i=3;i<p.length;i+=4)if(p[i])++n;return n;
  });
  assert.ok(targetPixels>0&&targetPixels<counts.painted[counts.lastLevel]/2,'Final target scale should render a smaller, nonempty mesh');
  await page.locator('.controls [data-scale="enlarged"]').click();
  let runtimeChecked=false;
  if(await page.locator('[data-level-mode="runtime"]').count()){
    await page.locator('[data-level-mode="runtime"]').click();
    const expected=await page.evaluate(()=>window.blitzBoard.data.assets.reduce((n,a)=>n+a.runtime_levels.length,0));
    assert.equal(await page.locator('.lod-card:visible').count(),expected);
    await page.locator('[data-level-mode="all"]').click();
    assert.equal(await page.locator('.lod-card:visible').count(),counts.expectedCards);
    runtimeChecked=true;
  }
  assert.ok(!await page.evaluate(()=>Boolean(window.blitzBoard.data.round4))||runtimeChecked);

  await page.locator('.lod-card[data-asset="2"][data-lod="'+counts.lastLevel+'"]').click();
  assert.equal(await page.locator('#inspector').evaluate(d=>d.open),true);
  assert.equal(await page.locator('#selected-triangles').textContent(),counts.expectedSelected.toLocaleString('en-US')+' triangles');
  await page.locator('[data-level="0"]').click();
  assert.equal(await page.locator('#selected-triangles').textContent(),counts.expectedSource.toLocaleString('en-US')+' triangles');
  const sameSource=await page.evaluate(()=>document.getElementById('source-view').toDataURL()===document.getElementById('selected-view').toDataURL());
  assert.ok(sameSource,'Comparing LOD 0 against itself must produce identical images');
  await page.locator('[data-level="'+counts.lastLevel+'"]').click();
  const beforeRotate=await page.locator('#selected-view').evaluate(c=>c.toDataURL());
  await page.locator('#rotate-right').click();
  await page.waitForFunction(before=>document.getElementById('selected-view').toDataURL()!==before,beforeRotate);
  await page.locator('#inspect-reset').click();
  await page.locator('#inspector [data-mode="wire"]').click();
  await page.locator('#inspector').screenshot({path:output+'/stool-comparison.png'});
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#inspector').evaluate(d=>d.open),false);
  await page.locator('.controls [data-mode="clay"]').click();
  await page.locator('.lod-card[data-asset="1"][data-lod="'+counts.lastLevel+'"]').click();
  await page.locator('#inspector [data-mode="silhouette"]').click();
  await page.locator('#inspector').screenshot({path:output+'/tree-comparison.png'});
  await page.keyboard.press('Escape');
  await page.locator('.controls [data-mode="clay"]').click();

  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(200);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth),false,'Mobile layout must not overflow horizontally');
  assert.equal(await page.locator('.lod-card').count(),counts.expectedCards);
  await page.screenshot({path:output+'/mobile-preview.png',fullPage:true});
  let catalogChecked=false;
  if(counts.catalog){
    const catalog=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
    catalog.on('pageerror',error=>failures.push(error.message));catalog.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
    await catalog.goto(new URL('catalog.html',pathToFileURL(input)).href);
    await catalog.waitForFunction(()=>document.documentElement.dataset.ready==='true');
    const expected=await catalog.evaluate(()=>window.foliageCatalog.data.targets);
    const total=Object.values(expected).reduce((a,b)=>a+b,0);assert.equal(await catalog.locator('.card').count(),total);
    assert.ok(await catalog.evaluate(()=>[...document.querySelectorAll('.card img')].every(i=>i.complete&&i.naturalWidth>0)));
    assert.ok(await catalog.evaluate(()=>[...document.querySelectorAll('.card .state')].every(s=>s.textContent.includes('Source preview')&&s.textContent.includes('original source data'))));
    assert.equal(await catalog.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await catalog.screenshot({path:output+'/catalog.png',fullPage:true});
    const height=await catalog.evaluate(()=>document.documentElement.scrollHeight);await catalog.emulateMedia({media:'screen'});
    await catalog.pdf({path:output+'/catalog.pdf',width:'1440px',height:height+'px',printBackground:true,margin:{top:0,bottom:0,left:0,right:0}});
    for(const [category,count] of Object.entries(expected)){await catalog.locator('button[data-category="'+category+'"]').click();assert.equal(await catalog.locator('.card:visible').count(),count);}
    await catalog.locator('button[data-category="all"]').click();assert.equal(await catalog.locator('.card:visible').count(),total);
    await catalog.setViewportSize({width:390,height:844});assert.equal(await catalog.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    catalogChecked=true;await catalog.close();
  }
  assert.deepEqual(requests,[],'The board must not fetch external scripts, models or fonts');
  assert.deepEqual(failures,[],'The board must have no browser errors');
  const record={
    browser:await browser.version(), desktop:{width:1600,height:counts.height},
    mobile_width:390, examples:counts.assets, lod_tiles:counts.expectedCards, all_tiles_painted:true,
    runtime_compaction_ui_checked:runtimeChecked,
    bake_seconds_and_vertex_modes_checked:chainMetadata,
    foliage_chains:counts.foliageChains,foliage_source_gallery:false,foliage_catalog_checked:catalogChecked,
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
