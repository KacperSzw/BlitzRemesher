// Export reproducible CSV and a standalone SVG with gnuplot 6.
import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..'),dir=resolve(root,'research/chain-search');
const report=JSON.parse(await fs.readFile(dir+'/analysis.json','utf8'));
const runs=report.runs.filter(r=>r.complete&&r.split==='development'&&r.preset==='coverage'&&r.method!=='topology-4');
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const balanced=(rows,fn)=>mean([...new Set(rows.map(r=>r.category))].map(c=>mean(rows.filter(r=>r.category===c).map(fn))));
const names={'baseline':'Incumbent','legacy-2':'Legacy 2x proposals','legacy-4':'Legacy 4x proposals','graph-2':'Graph: one pass','graph-4':'Graph: three passes'};
const colors={'baseline':'#7c8581','legacy-2':'#bc954e','legacy-4':'#b25c36','graph-2':'#157f69','graph-4':'#376cbb'};
let csv='run,pass,bake_seconds,chain_retained_percent,candidate_evaluations,audit_evaluations\n';
let script="set terminal svg size 1440,580 font 'Arial,12' background rgb '#f5f4ef'\nset output 'progress.svg'\nset multiplot layout 1,2 title 'Frozen eight-asset coverage pilot: reduction versus bake work' font ',19'\nset border 3 lc rgb '#b8c3bd'\nset tics nomirror textcolor rgb '#48544e'\nset grid ytics lc rgb '#dde2dd'\nset key bottom left\nset xlabel 'Summed elapsed bake seconds (shared workstation)'\nset ylabel 'Category-balanced chain triangles retained (%)'\nset xrange [0:*]\n";
const progressPlots=[],levelPlots=[];
for(const r of runs){
 const passes=Math.min(...r.rows.map(a=>a.progress.length));
 const progress=passes?Array.from({length:passes},(_,i)=>({pass:i,seconds:r.rows.reduce((s,a)=>s+a.progress[i].seconds,0),
  retention:100*balanced(r.rows,a=>a.progress[i].triangle_total/((a.triangles.length-1)*a.triangles[0])),
  calls:r.rows.reduce((s,a)=>s+a.progress[i].candidate_evaluations,0),audits:r.rows.reduce((s,a)=>s+a.progress[i].audit_evaluations,0)})):
 [{pass:0,seconds:r.rows.reduce((s,a)=>s+a.generation_seconds,0),retention:100*r.chain_ratio,calls:r.rows.reduce((s,a)=>s+a.work,0),audits:null}];
 for(const p of progress)csv+=`${r.id},${p.pass},${p.seconds},${p.retention},${p.calls},${p.audits??''}\n`;
 await fs.writeFile(`${dir}/progress-${r.method}.dat`,progress.map(p=>`${p.seconds} ${p.retention}`).join('\n')+'\n');
 await fs.writeFile(`${dir}/levels-${r.method}.dat`,Array.from({length:r.metadata.config.levels},(_,i)=>{
  const c=r.metadata.config,pixels=c.base_pixels*(c.last_pixels/c.base_pixels)**(i/(c.levels-1));
  return `${pixels} ${100*balanced(r.rows,a=>a.triangles[i]/a.triangles[0])}`;
 }).join('\n')+'\n');
 progressPlots.push(`'progress-${r.method}.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '${colors[r.method]}' title '${names[r.method]}'`);
 levelPlots.push(`'levels-${r.method}.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '${colors[r.method]}' title '${names[r.method]}'`);
}
script+='plot '+progressPlots.join(', \\\n')+'\n';
script+="set xlabel 'Scheduled screen diameter (pixels, logarithmic axis)'\nset ylabel 'Triangles retained per scheduled level (%)'\nset logscale x\nset xrange [512:16]\nset yrange [0:105]\nset xtics (512,256,128,64,32,16)\nset key top right\nplot "+levelPlots.join(', \\\n')+'\nunset multiplot\n';
await fs.writeFile(dir+'/progress.csv',csv);await fs.writeFile(dir+'/progress.gnuplot',script);
execFileSync(process.env.GNUPLOT??'gnuplot',['progress.gnuplot'],{cwd:dir,stdio:'inherit'});
console.log('Wrote progress.svg and its data. Different runs are individual timing observations; within-run points are complete incumbents.');
