// One audit worker. Each asset has its own 50-minute checkpoint deadline.
import fs from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
const root=resolve(import.meta.dirname,'..');
const [id,start='0',stride='1',experiment='research/chain-search',rotation='0xA1172026']=process.argv.slice(2);
if(!/^(chain-search|appearance|density)-[\w-]+$/.test(id??'')||!/^\d+$/.test(start)||!/^\d+$/.test(stride)||+stride<1)throw Error('Usage: RUN_ID [START_INDEX [STRIDE]]');
const read=async p=>JSON.parse(await fs.readFile(resolve(root,p),'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(resolve(root,p))).digest('hex');
const meta=await read(`research/runs/${id}/metadata.json`),pilot=await read('research/pilot.json');
const seed=Number(rotation);if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw Error('Invalid seed');
if(seed===meta.config.audit_views.seed)throw Error('Independent audit must rotate the configured cameras');
for(let i=+start;i<pilot.assets.length;i+=+stride) {
 const asset=pilot.assets[i].id,output=`${experiment}/audits/${id}/${asset}.json`;
 let previous;try{previous=await read(output);}catch(e){if(e.code!=='ENOENT')throw e;}
 if(previous?.complete) {
  if(previous.run_sha256!==meta.run_sha256||previous.chain_bin_sha256!==await hash(`research/runs/${id}/meshes/${asset}/chain.bin`)||previous.chain_gltf_sha256!==await hash(`research/runs/${id}/meshes/${asset}/chain.gltf`)||previous.row_sha256!==await hash(`research/runs/${id}/rows/${asset}.json`)||previous.audit.seed!==seed||previous.first_level!==1)throw Error('Mismatched audit checkpoint');
  continue;
 }
 await new Promise((yes,no)=>{const c=spawn(resolve(root,'build/release/blitz-tail-audit'),[`research/runs/${id}`,asset,output,rotation,'256','1'],{cwd:root,stdio:'inherit'});
  c.on('error',no);c.on('close',code=>code===0?yes():no(Error(`Audit process failed for ${asset}: ${code}`)));});
}
