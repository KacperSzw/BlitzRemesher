import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const dir='research/hybrid/opaque';await fs.mkdir(dir,{recursive:true});
const meta=await read('research/runs/hybrid-pilot-v5-mixed-auto-b8/metadata.json');
const binary='build/hybrid/blitz',hash=createHash('sha256').update(await fs.readFile(binary)).digest('hex');
if(hash!==meta.binary_sha256)throw Error('Binary differs from frozen pilot');
const stamp=dir+'/build.json';await fs.writeFile(stamp,JSON.stringify({binary_sha256:hash,source_tree_sha256:meta.source_tree_sha256,git_revision:meta.git_revision},null,2)+'\n');
for(const variant of ['core','preset']){
 const config=await read('research/configs/pilot-coupled.json');delete config.output;delete config.chain;config.triangle_overhead_bps=500;
 if(variant==='preset')config.research=(await read('research/hybrid/preset.json')).research;
 const path=dir+'/'+variant+'.json';await fs.writeFile(path,JSON.stringify(config,null,2)+'\n');
 await new Promise((resolve,reject)=>{const child=spawn(binary,['bench','research/pilot.json',path,`research/runs/opaque-hybrid-v5-${variant}`,'--build-stamp',stamp,'--minutes','50'],{stdio:'inherit'});child.on('error',reject);child.on('close',code=>code===0?resolve():reject(Error('Opaque exit '+code)));});
}
