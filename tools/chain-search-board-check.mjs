// BLITZ_PLAYWRIGHT_MODULE and BLITZ_CHROMIUM can point to local installations.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(import.meta.dirname,'..');
const {chromium}=await import(process.env.BLITZ_PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:process.env.BLITZ_CHROMIUM,headless:true,
 args:['--enable-unsafe-swiftshader','--use-angle=swiftshader','--renderer-process-limit=2','--num-raster-threads=2']});
const errors=[],requests=[];
try {
 const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 await context.route(/^https?:/,route=>{requests.push(route.request().url());return route.abort();});
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(resolve(root,'examples/reduction-board/index.html')).href);
 const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');await ready();
 const manifest=JSON.parse(await fs.readFile(resolve(root,'examples/reduction-board/manifest.json'),'utf8'));
 const observations=await page.evaluate(async()=>{
  const {data,state,redraw,selectPreset,payload}=window.blitzReductionBoard,checks=[];
  for(const preset of ['coverage','appearance','strict']) {
   state.preset=preset;document.getElementById('preset').value=preset;await selectPreset();
   for(const [index,r] of data.runs.entries())if(r.preset===preset) {
    state.run=index;document.getElementById('run').value=index;await redraw();
    const rows=[...document.querySelectorAll('.asset')];if(rows.length!==data.assets.length)throw Error('Unstable row count');
    for(const [i,a]of r.assets.entries())if(!a.unavailable) {
     if(rows[i].dataset.asset!==String(i))throw Error('Unstable asset order');
     const counts=[];
     for(const l of a.nodes){let triangles=0;for(const p of a.meshes[l.mesh].primitives){
      const indices=await payload(p.indices),positions=await payload(p.position);
      for(const id of indices)if(id>=positions.length/3)throw Error('Invalid preview index');
      triangles+=indices.length/3;
      if(p.normal&&(await payload(p.normal)).length!==positions.length)throw Error('Stored normal mismatch');
     }counts.push(triangles);}
     if(counts.some((n,i)=>n!==a.result.lods[i].triangles))throw Error('Displayed triangle count differs from geometry');
     if(a.inspection.levels.some(l=>l.cameras.length!==r.config.audit_views.orthographic+r.config.audit_views.perspective))throw Error('Camera contract mismatch');
     if(a.qualified&&(r.preset!=='appearance'||!a.dense?.passed||!a.dense?.complete||a.dense?.first_level!==1))throw Error('Invalid qualification');
     const labels=[...rows[i].querySelectorAll('.slot')].map(b=>b.textContent.replace(/\s+/g,''));
     if(labels.some((label,j)=>label!==('LOD'+j+counts[j].toLocaleString('en-US'))))throw Error('Scheduled labels mismatch');
    }
    checks.push({run:r.id,rows:rows.length});
   }
  }
  state.preset='coverage';document.getElementById('preset').value='coverage';await selectPreset();
  return checks;
 });
 assert.equal(observations.length,manifest.runs.length);
 for(let i=0;i<8;i++) {await page.locator('#level').fill(String(i));await page.locator('#level').dispatchEvent('input');await ready();assert.equal(await page.locator('#level-label').textContent(),'LOD '+i);}
 const pixels=async()=>(await page.locator('canvas[data-asset="1"]').evaluateAll(cs=>cs.map(c=>c.toDataURL()))).map(s=>createHash('sha256').update(s).digest('hex'));
 const before=await pixels(),canvas=page.locator('canvas[data-asset="1"][data-source="true"]');
 await canvas.scrollIntoViewIfNeeded();
 const box=await canvas.boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+35,box.y+box.height/2+12);await page.mouse.up();await ready();
 const rotated=await pixels();assert.notEqual(before[0],rotated[0]);assert.notEqual(before[1],rotated[1]);
 for(const mode of ['wire','silhouette','overlay','shaded']){await page.selectOption('#display',mode);await ready();if(mode==='wire')assert.notDeepEqual(await pixels(),rotated);}
 for(const camera of ['source-px','adjacent-px','source-area','adjacent-area','free']){await page.selectOption('#camera',camera);await ready();}
 await page.locator('#scale').click();await ready();
 const extents=await page.locator('canvas').evaluateAll(cs=>cs.map(c=>{
  const p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let loX=c.width,hiX=-1,loY=c.height,hiY=-1;
  for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++)if(p[(y*c.width+x)*4+3]){loX=Math.min(x,loX);hiX=Math.max(x,hiX);loY=Math.min(y,loY);hiY=Math.max(y,hiY);}
  return {width:hiX-loX+1,height:hiY-loY+1,backing:c.width,css:c.clientWidth};
 }));
 for(const e of extents){assert.ok(e.width>0&&e.width<=18&&e.height>0&&e.height<=18,JSON.stringify(e));assert.ok(Math.abs(e.backing-e.css)<=1);}
 await page.locator('#scale').click();await ready();await page.locator('#runtime').click();await ready();
 assert.ok(await page.evaluate(()=>[...document.querySelectorAll('.slot.duplicate')].every(e=>e.hidden)));
 await page.locator('#runtime').click();await ready();
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:resolve(root,'examples/reduction-board/preview.png')});
 await page.setViewportSize({width:390,height:844});await ready();
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile overflow');
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 const report={complete:true,offline:true,runs:observations,checks:['geometry counts and bounds','stored normal streams','stable row order','scheduled and runtime slots','matched rotation','four display modes','five camera modes','actual target pixel size','mobile layout'],page_errors:errors,network_requests:requests,browser:browser.version()};
 await fs.writeFile(resolve(root,'research/chain-search/board-check.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
}finally{await browser.close();}
