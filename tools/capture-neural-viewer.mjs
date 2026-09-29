// Verify the visual comparison and capture the actual saved meshes.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url),{chromium}=require(process.env.BLITZ_PLAYWRIGHT_MODULE??'playwright');
const dir=path.resolve(process.argv[2]??fileURLToPath(new URL('../research/neural/viewer',import.meta.url)));
const browser=await chromium.launch({headless:true,...(process.env.BLITZ_CHROMIUM?{executablePath:process.env.BLITZ_CHROMIUM}:{}),args:['--enable-unsafe-swiftshader','--use-angle=swiftshader','--renderer-process-limit=2','--num-raster-threads=2']});
const errors=[],requests=[],checks=[];
try{
  const page=await browser.newPage({viewport:{width:1440,height:980},deviceScaleFactor:1});
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  await page.goto(pathToFileURL(dir+'/index.html').href);await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  const assets=await page.evaluate(()=>window.neuralViewer.data.assets.map(a=>({id:a.id,levels:a.learned.levels.length})));
  for(let asset=0;asset<assets.length;asset++){
    await page.locator('button[data-asset="'+asset+'"]').click();
    for(let level=0;level<assets[asset].levels;level++){
      await page.locator('button[data-level="'+level+'"]').click();
      const result=await page.evaluate(()=>{const {data,state}=window.neuralViewer,a=data.assets[state.asset];return [...document.querySelectorAll('.frame canvas')].map(c=>{const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let painted=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])painted++;return {method:c.dataset.method,count:Number(c.dataset.triangles),expected:c.dataset.method==='source'?a.learned.levels[0].triangles:a[c.dataset.method].levels[state.level].triangles,painted,png:c.toDataURL()};});});
      for(const r of result){assert.equal(r.count,r.expected);assert.ok(r.painted>100);}
      if(level===0){assert.equal(result[0].png,result[1].png);assert.equal(result[0].png,result[2].png);}
    }
    const images=[];
    for(const mode of ['wire','shaded','changes','normals','uv']){
      await page.locator('button[data-mode="'+mode+'"]').click();
      images.push(await page.locator('.network .frame canvas').evaluate(c=>c.toDataURL()));
      if(['wire','shaded','changes'].includes(mode))await page.screenshot({path:dir+'/'+assets[asset].id+'-'+mode+'.png',fullPage:true});
      if(mode==='changes')await page.locator('#comparison').screenshot({path:dir+'/'+assets[asset].id+'-comparison.png'});
    }
    assert.equal(new Set(images).size,5,'The view modes must produce different images');
    await page.locator('button[data-mode="shaded"]').click();
    const before=await page.locator('.network .frame canvas').evaluate(c=>c.toDataURL());await page.locator('#right').click();
    assert.notEqual(await page.locator('.network .frame canvas').evaluate(c=>c.toDataURL()),before);
    await page.locator('#reset').click();assert.equal(await page.locator('.network .frame canvas').evaluate(c=>c.toDataURL()),before);
    const target=await page.locator('.network .target-row canvas').evaluate(c=>{const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])n++;return n;});
    assert.ok(target>0&&target<400,'The final target preview must retain its small pixel footprint');
    checks.push({asset:assets[asset].id,levels:assets[asset].levels,triangle_counts_match:true,lod0_identical:true,modes_distinct:true,rotation_and_reset:true,target_painted_pixels:target});
  }
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:dir+'/mobile-preview.png',fullPage:true});
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  const record={browser:await browser.version(),checks,network_requests:requests.length,errors,mobile_overflow:false};fs.writeFileSync(dir+'/browser-checks.json',JSON.stringify(record,null,2)+'\n');console.log(JSON.stringify(record,null,2));
}finally{await browser.close();}
