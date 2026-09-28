import fs from 'node:fs/promises';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const root='research/vegetation/board';await fs.mkdir(root,{recursive:true});
const original=await read('research/foliage/board/examples.json'),plan=await read('research/foliage/chains.json'),report=await read('research/vegetation/measurements.json');
const run=(variant,mode)=>`vegetation-matrix-v3-${variant}-${mode}-b8`;
const row=async(variant,mode,id)=>read(`research/runs/${run(variant,mode)}/rows/${id}.json`);
const examples=original.examples.slice(0,4),comparisons=[];
async function example(asset,mode,variant){
 const e={...asset,run:run(variant,mode),output:mode,name:asset.name+' · '+(variant==='legacy'?'before':'candidate')+' / '+(mode==='reuse'?'shared':'rebuilt'),note:'8 proposals per level; identical pixel limits. Geometry-only, without opacity or shading.'};
 try{
  const d=await read(`research/vegetation/dense/${variant}-${mode}-${asset.id}.json`),r=await row(variant,mode,asset.id);
  if(d.run!==run(variant,mode)||d.run_sha256!==r.run_sha256||d.chain_bin_sha256!==r.output_sha256||d.chain_gltf_sha256!==r.gltf_sha256||d.audit.seed!==0xB1172031)throw Error('Dense audit does not match displayed chain: '+asset.id);
  e.dense_passed=d.passed;
 }catch(error){if(error.code!=='ENOENT')throw error;}
 examples.push(e);
}
for(const id of ['ph_fern_02_clump_1','loaf_tree_1']){const a=plan.examples.find(a=>a.id===id);for(const mode of ['reuse','rebuild'])for(const variant of ['legacy','candidate'])await example(a,mode,variant);}
for(const a of plan.examples)if(!['ph_fern_02_clump_1','loaf_tree_1'].includes(a.id))for(const mode of ['reuse','rebuild'])await example(a,mode,'candidate');
for(const a of plan.examples)for(const mode of ['reuse','rebuild']){const before=await row('legacy',mode,a.id),after=await row('candidate',mode,a.id);if(!before.complete||!after.complete)throw Error('Incomplete board input');
 comparisons.push({id:a.id,name:a.name,mode:mode==='reuse'?'shared vertices':'rebuilt vertices',budget:8,before_final:before.result.lods.at(-1).triangles,after_final:after.result.lods.at(-1).triangles,before_tail:before.result.lods.slice(-3).reduce((s,l)=>s+l.triangles,0),after_tail:after.result.lods.slice(-3).reduce((s,l)=>s+l.triangles,0),before_seconds:before.generation_seconds,after_seconds:after.generation_seconds,before_area:before.result.lods.at(-1).source.changed_area,after_area:after.result.lods.at(-1).source.changed_area});}
const reduction=mode=>{const a=report.runs.find(r=>r.name===run('legacy',mode)),b=report.runs.find(r=>r.name===run('candidate',mode));if(!a?.complete||!b?.complete)throw Error('Incomplete summary');return (100*(1-b.retention.last_three/a.retention.last_three)).toFixed(1)+'%';};
const dense=examples.filter(e=>e.dense_passed!==undefined),passed=dense.filter(e=>e.dense_passed).length;
const data={intro:'Six nature assets in both vertex modes. Fern and Broadleaf tree have paired before/after chains. Every tile contains the actual baked geometry; use target scale to inspect the distant LODs.',audit_note:'The new comparisons use the same eight-proposal budget and sampled pixel gates. Dense tail checks are reported separately and can fail. Coverage change is a filled-geometry diagnostic, not an opacity or shading score. Rows 01–04 are archived opaque examples.',stats:[{label:'Shared · tail improvement',value:reduction('reuse'),detail:'Lower retained ratio vs control · same six assets, category balance and budget'},{label:'Rebuilt · tail improvement',value:reduction('rebuild'),detail:'Lower retained ratio vs control · same six assets, category balance and budget'},{label:'Nature research collection',value:'40',detail:'28 development models · 12 validation models · families kept together'},{label:'Dense tail checks passed',value:`${passed}/${dense.length}`,detail:'706-camera set · source and adjacent checks · 8× to 32× sampling'}],comparisons};
await fs.writeFile('research/vegetation/board-data.json',JSON.stringify(data,null,2)+'\n');
await fs.writeFile(root+'/examples.json',JSON.stringify({...original,date:'28 September 2026',examples,vegetation:'research/vegetation/board-data.json'},null,2)+'\n');
console.log(examples.length+' board chains, '+comparisons.length+' matched comparisons');
