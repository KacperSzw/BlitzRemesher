import fs from 'node:fs/promises';import {createHash} from 'node:crypto';import {spawn} from 'node:child_process';import {resolve} from 'node:path';
process.chdir(resolve(import.meta.dirname,'..'));
const [version,stage,cohort='development']=process.argv.slice(2);
if(!/^v\d+$/.test(version??'')||!['screen','ordering','attributes','merged','merged-attributes','shared'].includes(stage)||!['development','validation','stool','trunk'].includes(cohort))throw Error('Usage: VERSION STAGE [development|validation|stool|trunk]');
const smoke=['stool','trunk'].includes(cohort),binary=`build/density/${version}/blitz`,stamp=`research/density/builds/${version}.json`;
const build=JSON.parse(await fs.readFile(stamp));if(createHash('sha256').update(await fs.readFile(binary)).digest('hex')!==build.binary_sha256)throw Error('Frozen binary hash mismatch');
const manifest={development:'research/pilot.json',validation:'research/corpus.json',stool:'research/chain-search/smoke.json',trunk:'research/appearance/trunk-smoke.json'}[cohort];
const id=`density-${version}-${stage}${smoke?'-smoke-'+cohort:cohort==='validation'?'-validation':''}`;
await new Promise((yes,no)=>{const c=spawn(binary,['bench',manifest,`research/density/configs/${stage}${smoke?'-smoke':''}.json`,`research/runs/${id}`,'--build-stamp',stamp,'--split',cohort==='validation'?'validation':'development','--minutes',smoke?'10':'50'],{stdio:'inherit'});c.on('error',no);c.on('close',(code,signal)=>code===0?yes():no(Error(`Run exited ${code??signal}; partial rows retained`)));});
