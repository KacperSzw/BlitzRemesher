// Check the explanation against raw evidence, exercise its controls, and capture it.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url),{chromium}=require(process.env.BLITZ_PLAYWRIGHT_MODULE??'playwright');
const root=fileURLToPath(new URL('..',import.meta.url));
const dir=path.resolve(process.argv[2]??path.join(root,'research/neural/viewer'));
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const data=read(dir+'/learning-loop-data.json'),manifest=read(dir+'/learning-loop-manifest.json');
for(const f of manifest.files)assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,f.path))).digest('hex'),f.sha256);
const evidence=path.join(root,'research/neural/evidence/action-screening-v3');
const outcome=read(evidence+'/final/outcome.json'),result=read(evidence+'/final/result.json');
const reports=[1,2].map(i=>read(evidence+`/final/pilot-${i}/report.json`));
const rawRock=reports[1].runs.find(r=>r.name==='learned-0').rows.find(r=>r.id==='ph_moon_rock_02');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
near(data.charts.rental.seconds,outcome.rental_minutes*60);
near(data.charts.rental.segments.find(s=>s.id==='training').seconds,result.stages.flatMap(s=>s.health).reduce((n,h)=>n+h.seconds,0));
near(data.charts.rental.segments.find(s=>s.id==='evaluation').seconds,read(evidence+'/action-v2-staged/baseline-health.json').summary.seconds+reports.flatMap(r=>r.runs).reduce((n,r)=>n+r.wall_seconds,0));
near(data.charts.rock.seconds,rawRock.seconds);
near(data.charts.rock.segments.find(s=>s.id==='audit').seconds,rawRock.neural.gpu_audit_seconds+rawRock.neural.gpu_confirmation_seconds);
near(data.charts.rock.segments.find(s=>s.id==='inference').seconds,rawRock.neural.inference_seconds);
for(const c of Object.values(data.charts)){assert.ok(c.segments.every(s=>s.seconds>=0));near(c.segments.reduce((n,s)=>n+s.seconds,0),c.seconds);}
const browser=await chromium.launch({headless:true,...(process.env.BLITZ_CHROMIUM?{executablePath:process.env.BLITZ_CHROMIUM}:{}),args:['--renderer-process-limit=2','--num-raster-threads=2']});
const errors=[],requests=[],links=new Set(),nodeChecks=[],chartChecks=[];
try{
  const page=await browser.newPage({viewport:{width:1440,height:1120},deviceScaleFactor:1});
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  await page.goto(pathToFileURL(dir+'/learning-loop.html').href);await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  assert.deepEqual(await page.evaluate(()=>window.neuralLoop.data),data);
  const collectLinks=async()=>{for(const href of await page.locator('a').evaluateAll(as=>as.map(a=>a.getAttribute('href'))))links.add(href);};
  for(const diagram of ['learning','audit']){
    await page.locator('[data-diagram="'+diagram+'"]').click();
    const nodes=await page.locator('[data-node]').evaluateAll(ns=>ns.map(n=>({id:n.dataset.node,title:n.querySelector('strong').textContent})));
    for(const n of nodes){await page.locator('[data-node="'+n.id+'"]').click();assert.equal(await page.locator('#node-detail h3').textContent(),n.title);assert.equal(await page.locator('[data-node][aria-pressed="true"]').count(),1);await collectLinks();}
    const first=page.locator('[data-node="'+nodes[0].id+'"]');await first.focus();await page.keyboard.press('Home');
    assert.equal(await page.evaluate(()=>window.neuralLoop.state.node),nodes[0].id);
    await page.keyboard.press('ArrowRight');assert.equal(await page.evaluate(()=>window.neuralLoop.state.node),nodes[1].id);
    assert.equal(await page.evaluate(()=>document.activeElement.dataset.node),nodes[1].id);
    await page.keyboard.press('End');assert.equal(await page.evaluate(()=>window.neuralLoop.state.node),nodes.at(-1).id);
    await page.locator('#previous-step').click();assert.equal(await page.evaluate(()=>window.neuralLoop.state.node),nodes.at(-2).id);
    await page.locator('#next-step').click();assert.equal(await page.evaluate(()=>window.neuralLoop.state.node),nodes.at(-1).id);
    const paths=await page.locator('#graph svg > path').evaluateAll(ps=>ps.map(p=>p.getAttribute('d')));
    assert.ok(paths.length>=nodes.length);assert.ok(paths.every(p=>!p.includes('NaN')&&!p.includes('undefined')));
    nodeChecks.push({diagram,nodes:nodes.length,selection:true,keyboard:true,arrows:paths.length});
  }
  for(const chart of ['rental','rock']){
    await page.locator('[data-chart="'+chart+'"]').click();
    const segments=data.charts[chart].segments;
    for(const s of segments){await page.locator('#chart-list [data-segment="'+s.id+'"]').click();assert.equal(await page.locator('#chart-detail h3').textContent(),s.name);assert.equal(await page.locator('#chart-detail .small-title').textContent(),s.kind);assert.equal(await page.locator('#chart-list [aria-pressed="true"]').count(),1);await collectLinks();}
    // The stack is proportional even when a tiny inference segment is below one pixel.
    const widths=await page.locator('#stack .segment').evaluateAll(bs=>bs.map(b=>parseFloat(b.style.width)));
    // CSSOM serializes percentage values at lower precision than the raw timings.
    widths.forEach((w,i)=>assert.ok(Math.abs(w-100*segments[i].seconds/data.charts[chart].seconds)<1e-4));
    const first=page.locator('#stack .segment').first();await first.click();assert.equal(await page.locator('#chart-detail h3').textContent(),segments[0].name);
    chartChecks.push({chart,segments:segments.length,proportional:true,selection:true});
  }
  await page.locator('#evidence summary').click();assert.equal(await page.locator('#evidence').getAttribute('open'),'');await collectLinks();await page.locator('#evidence summary').click();
  const localLinks=[...links].filter(s=>!/^https?:/.test(s));for(const href of localLinks)assert.ok(fs.existsSync(path.resolve(dir,href.split('#')[0])),'Missing local link: '+href);
  assert.ok([...links].filter(s=>/^https?:/.test(s)).every(s=>s.startsWith('https://')));
  await page.locator('[data-diagram="learning"]').click();await page.locator('[data-chart="rental"]').click();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:dir+'/learning-loop.png',fullPage:true});
  await page.locator('.section').first().screenshot({path:dir+'/learning-loop-detail.png'});
  await page.locator('[data-diagram="audit"]').click();await page.locator('[data-chart="rock"]').click();
  await page.screenshot({path:dir+'/learning-loop-audit.png',fullPage:true});
  await page.locator('.section').first().screenshot({path:dir+'/learning-loop-audit-detail.png'});
  for(const width of [900,390]){
    await page.setViewportSize({width,height:900});await page.waitForTimeout(100);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    for(const diagram of ['learning','audit']){
      await page.locator('[data-diagram="'+diagram+'"]').click();
      const boxes=await page.locator('[data-node]').evaluateAll(ns=>ns.map(n=>{const b=n.getBoundingClientRect();return {left:b.left,right:b.right,width:b.width};}));
      assert.ok(boxes.every(b=>b.left>=0&&b.right<=width&&b.width>0));
      await page.locator('[data-node]').first().click();assert.equal(await page.locator('#previous-step').isDisabled(),true);
    }
  }
  await page.locator('[data-diagram="learning"]').click();await page.locator('[data-chart="rental"]').click();await page.screenshot({path:dir+'/learning-loop-mobile.png',fullPage:true});
  // Check the link supplied by the original mesh viewer without loading its WebGL renderer.
  assert.ok(fs.readFileSync(dir+'/index.html','utf8').includes('href="learning-loop.html"'));
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  const checks={browser:await browser.version(),input_hashes_verified:manifest.files.length,raw_timing_cross_checks:true,node_checks:nodeChecks,chart_checks:chartChecks,local_links_verified:localLinks.length,viewports:[1440,900,390],horizontal_overflow:false,network_requests:requests.length,errors};
  fs.writeFileSync(dir+'/learning-loop-checks.json',JSON.stringify(checks,null,2)+'\n');console.log(JSON.stringify(checks,null,2));
}finally{await browser.close();}
