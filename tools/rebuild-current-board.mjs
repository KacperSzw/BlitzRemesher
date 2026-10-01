// Rebuild every selected chain with its frozen settings, then embed the results.
import fs from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {resolve} from 'node:path';
import {entries} from './current-board-assets.mjs';

const root=resolve(import.meta.dirname,'..');
const run=promisify(execFile);
const requested=new Set(process.argv.slice(2));
for(const id of requested)if(!entries.some(e=>e.id===id))throw Error('Unknown board asset: '+id);
for(const entry of entries) {
  if(requested.size&&!requested.has(entry.id))continue;
  const {stdout}=await run(resolve(root,'build/release/blitz'),
    ['simplify',entry.input,'--config',entry.config,'--out',entry.dir],
    {cwd:root,maxBuffer:8*1024*1024});
  const result=JSON.parse(stdout);
  if(result.status!=='complete')throw Error(entry.id+' bake was '+result.status);
  await fs.writeFile(resolve(root,entry.result),stdout);
  console.log(entry.id+': '+result.lods.at(-1).triangles+' final triangles, '+result.seconds.toFixed(2)+' s');
}
await run('node',['tools/current-board.mjs'],{cwd:root});
console.log('Current Board regenerated with '+entries.length+' assets.');
