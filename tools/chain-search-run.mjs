// Frozen, resumable research runs. Use one process for timing comparisons.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const [version='v2',preset='coverage',method='graph-4',split='development']=process.argv.slice(2);
if(!/^v\d+$/.test(version)||!['coverage','appearance','strict'].includes(preset)||!['baseline','legacy-2','legacy-4','graph-2','graph-4','topology-4'].includes(method)||!['development','validation'].includes(split))throw Error('Usage: VERSION PRESET METHOD [development|validation]');
const legacy=method.startsWith('legacy')||method==='baseline';
const binary=`build/chain-search/${legacy?'baseline':version}/blitz`;
const stamp=`research/chain-search/builds/${legacy?'baseline':`candidate-${version}`}.json`;
const configuration=`research/chain-search/configs/${preset}-${method}.json`;
const expected=JSON.parse(await fs.readFile(resolve(root,stamp),'utf8'));
const sha=createHash('sha256').update(await fs.readFile(resolve(root,binary))).digest('hex');
if(sha!==expected.binary_sha256)throw Error('Frozen executable hash mismatch');
const manifest=split==='validation'?'research/corpus.json':'research/pilot.json';
const id=`chain-search-${legacy?'':version+'-'}${preset}-${method}${split==='validation'?'-validation':''}`;
const output=`research/runs/${id}`;
await fs.mkdir(resolve(root,output),{recursive:true});
await new Promise((yes,no)=>{
 const child=spawn(resolve(root,binary),['bench',manifest,configuration,output,'--build-stamp',stamp,'--split',split,'--minutes','50'],{cwd:root,stdio:'inherit'});
 child.on('error',no);child.on('close',code=>code===0?yes():no(Error(`Run ${id} exited ${code}; keep its checkpoint and incomplete status.`)));
});
