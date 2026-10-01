// Restore only the frozen external inputs needed to rebuild the Current Board.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {dirname,resolve} from 'node:path';
import {entries} from './current-board-assets.mjs';

const root=resolve(import.meta.dirname,'..');
const run=promisify(execFile);
const read=async path=>JSON.parse(await fs.readFile(resolve(root,path),'utf8'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const catalogs=await Promise.all([read('research/pilot.json'),read('research/corpus.json')]);
const assets=new Map(catalogs.flatMap(c=>c.assets).map(a=>[a.id,a]));
let restored=0;
for(const entry of entries) {
  if(!entry.input.startsWith('data/'))continue;
  const asset=assets.get(entry.id);
  if(!asset||asset.split!=='development'||!asset.opaque||asset.path!==entry.input)
    throw Error('Missing frozen development asset: '+entry.id);
  for(const file of asset.files) {
    if(!file.path.startsWith('data/')||!file.url)throw Error('Invalid frozen file: '+entry.id);
    const path=resolve(root,file.path);
    try {
      if(hash(await fs.readFile(path))===file.sha256)continue;
      throw Error('Existing source hash mismatch: '+file.path);
    }catch(error) {
      if(error.code!=='ENOENT')throw error;
    }
    await fs.mkdir(dirname(path),{recursive:true});
    const temporary=path+'.part';
    try {
      await run('curl',['--fail','--location','--retry','3','--silent','--show-error','--output',temporary,file.url]);
      const bytes=await fs.readFile(temporary);
      if(bytes.length!==file.bytes||hash(bytes)!==file.sha256)throw Error('Frozen source hash mismatch: '+file.path);
      await fs.rename(temporary,path);
      ++restored;
    }catch(error) {
      await fs.rm(temporary,{force:true});
      throw error;
    }
  }
}
console.log(`Restored ${restored} frozen board files; ${entries.length} example assets available.`);
