#!/nix/store/i1rgc6zsrxbxgfz97vbczz7cxcg71p5s-nodejs-slim-24.19.0/bin/node
const{spawnSync}=require('node:child_process');const env=JSON.parse(require('node:fs').readFileSync('/tmp/blitz-gdb15/env.json'));const r=spawnSync('/tmp/blitz-gdb15/usr/bin/gdb',process.argv.slice(2),{env:{...process.env,...env},stdio:'inherit'});process.exit(r.status??1);
