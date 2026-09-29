import fs from 'node:fs/promises';import {createHash} from 'node:crypto';import {spawn} from 'node:child_process';import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');process.chdir(root);
const [stage,cohort='development',version=stage==='screen'?'screen':'qem-v1']=process.argv.slice(2);
if(!['screen','ordering','attributes','position'].includes(stage)||!['development','validation','stool','trunk'].includes(cohort)||!/^[a-z0-9-]+$/.test(version))throw Error('Usage: STAGE [development|validation|stool|trunk] [FROZEN_VERSION]');
const binary=`build/appearance/${version}/blitz`,stamp=`research/appearance/builds/${version}.json`;
const expected=JSON.parse(await fs.readFile(stamp));if(createHash('sha256').update(await fs.readFile(binary)).digest('hex')!==expected.binary_sha256)throw Error('Frozen binary hash mismatch');
const manifest={development:'research/pilot.json',validation:'research/corpus.json',stool:'research/chain-search/smoke.json',trunk:'research/appearance/trunk-smoke.json'}[cohort];
const id=`appearance-${stage}${['stool','trunk'].includes(cohort)?'-smoke-'+cohort:cohort==='validation'?'-validation':''}`;
await new Promise((yes,no)=>{const child=spawn(binary,['bench',manifest,`research/appearance/configs/${stage}.json`,`research/runs/${id}`,'--build-stamp',stamp,'--split',cohort==='validation'?'validation':'development','--minutes',['stool','trunk'].includes(cohort)?'10':'50'],{stdio:'inherit'});child.on('error',no);child.on('close',(code,signal)=>code===0?yes():no(Error(`Run exited ${code??signal}; partial evidence retained`)));});
