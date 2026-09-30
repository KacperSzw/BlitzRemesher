import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const hash=f=>createHash('sha256').update(fs.readFileSync(f)).digest('hex');
export function validateExpanded({original,allowed,corpus,training,validation,trainingGroups,validationGroups}){
  const rows=new Map(corpus.assets.map(a=>[a.id,a]));
  if(rows.size!==corpus.assets.length)throw Error('Duplicate corpus ID');
  for(const a of original.assets)if(JSON.stringify(rows.get(a.id))!==JSON.stringify(a))throw Error('Original corpus asset or split changed');
  const allowedIds=new Set(allowed.assets.map(a=>a.id));
  const excluded=new Set(original.assets.filter(a=>!allowedIds.has(a.id)).map(a=>a.source_group));
  // Multiple variants in an allowed family are harmless, but an excluded
  // development/validation/holdout family may never enter the training set.
  const groups=new Set(),geometry=new Set();
  for(const [selection,count,split] of [[training,trainingGroups,'development'],[validation,validationGroups,'validation']]){
    if(selection.assets.length!==count)throw Error('Independent source-group count differs');
    for(const a of selection.assets){
      if(JSON.stringify(rows.get(a.id))!==JSON.stringify(a)||a.split!==split)throw Error('Selection differs from source corpus or split');
      if(groups.has(a.source_group)||geometry.has(a.geometry_sha256))throw Error('Source family or geometry leaks between selections');
      if(split==='development'&&excluded.has(a.source_group))throw Error('Excluded source group entered training');
      if(a.license!=='CC0-1.0'||!a.opaque||a.import_status!=='ok')throw Error('Unsupported asset counted');
      groups.add(a.source_group);geometry.add(a.geometry_sha256);
    }
  }
  return {training_groups:trainingGroups,validation_groups:validationGroups,original_assets_preserved:original.assets.length};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [directory]=process.argv.slice(2);if(!directory)throw Error('corpus.mjs CORPUS_DIRECTORY');
  const read=f=>JSON.parse(fs.readFileSync(f));const contract=read(directory+'/contract.json'),complete=read(directory+'/complete.json');
  const original=read('research/corpus.json'),allowed=read('research/neural/training-manifest.json');
  if(hash('research/corpus.json')!==contract.source_sha256||hash('research/neural/training-manifest.json')!==contract.selection_sha256)throw Error('Parent corpus checksum differs');
  for(const name of ['corpus','training','validation'])if(hash(directory+'/'+name+'.json')!==complete[name+'_sha256'])throw Error('Frozen expanded checksum differs');
  console.log(JSON.stringify(validateExpanded({original,allowed,corpus:read(directory+'/corpus.json'),training:read(directory+'/training.json'),validation:read(directory+'/validation.json'),trainingGroups:contract.training_groups,validationGroups:contract.validation_groups}),null,2));
}
