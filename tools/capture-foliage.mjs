// Render original source geometry for the collection contact sheet. This is
// presentation only: the sidecar alpha hookups do not enter reduction/SCORE.
import {createRequire} from 'node:module';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.BLITZ_PLAYWRIGHT_MODULE||'playwright');
const input=resolve(process.argv[2]||'build/foliage/previews.json');
const output=resolve(process.argv[3]||'research/foliage/previews');
const data=JSON.parse(await readFile(input,'utf8'));await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.BLITZ_CHROMIUM?{executablePath:process.env.BLITZ_CHROMIUM}:{}),args:['--enable-unsafe-swiftshader','--use-angle=swiftshader','--renderer-process-limit=2','--num-raster-threads=2']});
const failures=[],requests=[];
try{
  const page=await browser.newPage();page.on('pageerror',e=>failures.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  await page.setContent('<canvas id="view" width="480" height="360"></canvas>');
  await page.evaluate(async data=>{
    const canvas=document.getElementById('view'),gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true,alpha:false});if(!gl)throw Error('WebGL2 unavailable');
    const shader=(kind,source)=>{const s=gl.createShader(kind);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    const program=gl.createProgram();
    gl.attachShader(program,shader(gl.VERTEX_SHADER,`#version 300 es
      in vec3 position;in vec2 uv;uniform vec3 center;uniform float extent,yaw,pitch,flip;out vec3 world;out vec2 texcoord;
      void main(){vec3 p=(position-center)/extent;float c=cos(yaw),s=sin(yaw);p=vec3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z);c=cos(pitch);s=sin(pitch);p=vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);world=p;texcoord=vec2(uv.x,mix(uv.y,1.-uv.y,flip));gl_Position=vec4(p.x*.75,p.y,-p.z*.4,1.);}`));
    gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`#version 300 es
      precision highp float;in vec3 world;in vec2 texcoord;uniform sampler2D colorMap,opacityMap;uniform int channel;uniform float cutoff;uniform vec4 factor;out vec4 color;
      void main(){vec4 base=texture(colorMap,texcoord)*factor;vec4 mask=texture(opacityMap,texcoord);float a=channel==1?mask.r:mask.a;if(a*factor.a<cutoff)discard;vec3 n=normalize(cross(dFdx(world),dFdy(world)));float light=.48+.52*abs(dot(n,normalize(vec3(-.4,.7,1.))));color=vec4(base.rgb*light,1.);}`));
    gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
    const u=Object.fromEntries(['center','extent','yaw','pitch','flip','channel','cutoff','factor','colorMap','opacityMap'].map(n=>[n,gl.getUniformLocation(program,n)]));
    const textures=new Map(),imageChecks=[];
    for(const [id,url] of Object.entries(data.images)){
      const image=new Image();image.src=url;await image.decode();const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);gl.generateMipmap(gl.TEXTURE_2D);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);textures.set(id,t);
      const c=document.createElement('canvas');c.width=image.width;c.height=image.height;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,c.width,c.height).data;let alphaMin=255,alphaMax=0,redMin=255,redMax=0;
      for(let i=0;i<pixels.length;i+=4){alphaMin=Math.min(alphaMin,pixels[i+3]);alphaMax=Math.max(alphaMax,pixels[i+3]);redMin=Math.min(redMin,pixels[i]);redMax=Math.max(redMax,pixels[i]);}
      imageChecks.push({id,width:image.width,height:image.height,alphaMin,alphaMax,redMin,redMax});
    }
    const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer;
    gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.uniform1i(u.colorMap,0);gl.uniform1i(u.opacityMap,1);
    window.renderAsset=index=>{
      const asset=data.assets[index],low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];let triangles=0;const buffers=[];
      const meshes=asset.primitives.map(p=>{const v=new Float32Array(decode(p.vertices)),ix=new Uint32Array(decode(p.indices));triangles+=ix.length/3;
        for(let i=0;i<v.length;i+=5)for(let a=0;a<3;a++){low[a]=Math.min(low[a],v[i+a]);high[a]=Math.max(high[a],v[i+a]);}return {p,v,ix};});
      if(triangles!==asset.triangles)throw Error('Source triangle count mismatch: '+asset.id);
      const center=low.map((v,i)=>(v+high[i])/2);let radius=0;for(const {v} of meshes)for(let i=0;i<v.length;i+=5)radius=Math.max(radius,Math.hypot(v[i]-center[0],v[i+1]-center[1],v[i+2]-center[2]));
      gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.945,.945,.914,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniform3fv(u.center,center);gl.uniform1f(u.extent,radius*1.08);gl.uniform1f(u.yaw,asset.preview.yaw);gl.uniform1f(u.pitch,asset.preview.pitch);gl.uniform1f(u.flip,asset.preview.flip_v?1:0);
      for(const {p,v,ix} of meshes){const vb=gl.createBuffer(),ib=gl.createBuffer();buffers.push(vb,ib);gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,v,gl.STATIC_DRAW);for(const [name,size,offset] of [['position',3,0],['uv',2,12]]){const a=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,size,gl.FLOAT,false,20,offset);}gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,ix,gl.STATIC_DRAW);
        gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,textures.get(p.color));gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,textures.get(p.opacity));gl.uniform1i(u.channel,p.channel);gl.uniform1f(u.cutoff,p.cutoff);gl.uniform4fv(u.factor,p.factor);gl.drawElements(gl.TRIANGLES,ix.length,gl.UNSIGNED_INT,0);
      }
      const pixels=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);const error=gl.getError();if(error)throw Error('WebGL error '+error);let painted=0;
      for(let i=0;i<pixels.length;i+=4)if(Math.abs(pixels[i]-241)+Math.abs(pixels[i+1]-241)+Math.abs(pixels[i+2]-233)>10)painted++;
      buffers.forEach(b=>gl.deleteBuffer(b));return {id:asset.id,triangles,painted,png:canvas.toDataURL('image/png')};
    };
    window.imageChecks=imageChecks;
  },data);
  const records=[];
  for(let i=0;i<data.assets.length;i++){
    const r=await page.evaluate(i=>window.renderAsset(i),i);assert.ok(r.painted>40,'Source preview must be visible: '+r.id);const bytes=Buffer.from(r.png.split(',')[1],'base64');await writeFile(output+'/'+r.id+'.png',bytes);delete r.png;r.sha256=createHash('sha256').update(bytes).digest('hex');records.push(r);
  }
  const images=await page.evaluate(()=>window.imageChecks);
  // Every sidecar opacity binding must address a real varying alpha/R channel.
  for(const asset of data.assets)for(const p of asset.primitives)if(p.cutoff>0){const im=images.find(i=>i.id===p.opacity);assert.ok(p.channel?im.redMin<im.redMax:im.alphaMin<im.alphaMax,'Opacity channel must vary: '+asset.id);}
  assert.deepEqual(failures,[]);assert.deepEqual(requests,[]);
  await writeFile(output+'/checks.json',JSON.stringify({manifest_sha256:data.manifest_sha256,browser:await browser.version(),presentation_only:true,source_triangles_verified:true,opacity_channels_vary:true,network_requests:requests.length,errors:failures,images,assets:records},null,2)+'\n');
  console.log('Rendered '+records.length+' source models; decoded '+images.length+' textures; no network requests.');
}finally{await browser.close();}
