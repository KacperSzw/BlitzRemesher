#include "neural_vulkan_cuda.hpp"
#include "neural_vulkan.hpp"
#include "neural_raster_pixel.cuh"
#include "neural_cuda.cuh"
#include <cub/cub.cuh>
namespace blitz::neural {
DrawLayout draw_layout(DeviceMeshView m,NeuralVertexStorage storage,bool coverage_only){
    if(coverage_only){m.normals=nullptr;m.uv=nullptr;m.colors=nullptr;m.tangents=nullptr;}
    DrawLayout l{};size_t at=0;auto stream=[&](size_t bytes){at=(at+3)&~size_t(3);auto offset=at;at+=bytes;return offset;};
    l.position_stride=storage==NeuralVertexStorage::Float32?12:6;l.normal_stride=storage==NeuralVertexStorage::Packed?4:12;
    l.position=stream(size_t(m.vertices)*l.position_stride);
    l.normal=stream(size_t(m.normals?m.vertices:1)*l.normal_stride);
    l.uv=stream(m.uv?size_t(m.vertices)*(storage==NeuralVertexStorage::Packed?4:8):0);
    l.color=stream(size_t(m.colors?m.vertices:1)*4);
    l.tangent=stream(m.tangents?size_t(m.vertices)*(storage==NeuralVertexStorage::Packed?4:16):0);
    if(m.exact_position_bits&&storage!=NeuralVertexStorage::Float32){size_t words=(size_t(m.vertices)+31)/32;l.exact_bits=stream(words*4);l.exact_rank=stream(words*4);l.used=stream(words*4);l.exact_positions=stream((uint64_t(m.vertices)*m.exact_position_bps/10000)*sizeof(Vec3));}
    l.indices=stream(size_t(m.faces)*12);at=(at+255)&~size_t(255);l.faces=stream(size_t(m.faces)*4);at=(at+255)&~size_t(255);l.metadata=stream(sizeof(DrawMetadata));l.indirect=stream(5*sizeof(uint32_t));l.status=stream(10*sizeof(uint32_t));l.bytes=at;return l;
}
struct MinMax {Vec3 lo,hi;uint32_t invalid;};
struct CombineBounds {__device__ MinMax operator()(MinMax a,MinMax b){return {{fminf(a.lo.x,b.lo.x),fminf(a.lo.y,b.lo.y),fminf(a.lo.z,b.lo.z)},{fmaxf(a.hi.x,b.hi.x),fmaxf(a.hi.y,b.hi.y),fmaxf(a.hi.z,b.hi.z)},a.invalid|b.invalid};}};
__global__ void draw_bounds(DeviceMeshView m,DrawMetadata* out,bool packed,DrawLayout l,char* buffer){
    if(m.trial_status&&m.trial_status->invalid){if(threadIdx.x==0)*out={};return;}
    MinMax v{{INFINITY,INFINITY,INFINITY},{-INFINITY,-INFINITY,-INFINITY},0};
    for(uint32_t i=threadIdx.x;i<m.vertices;i+=blockDim.x){auto p=m.positions[i];v=CombineBounds{}(v,{p,p,0});
        if(!isfinite(p.x)||!isfinite(p.y)||!isfinite(p.z))v.invalid|=DrawNonfinite;
        if(m.fixed_quantization&&(p.x<m.quant_low.x||p.y<m.quant_low.y||p.z<m.quant_low.z||p.x>m.quant_low.x+m.quant_extent.x||p.y>m.quant_low.y+m.quant_extent.y||p.z>m.quant_low.z+m.quant_extent.z))v.invalid|=DrawPosition;
        if(packed&&m.uv){auto t=m.uv[i];if(!(t.x>=-8&&t.x<=8&&t.y>=-8&&t.y<=8))v.invalid|=DrawUv;}}
    __shared__ cub::BlockReduce<MinMax,256>::TempStorage temp;auto b=cub::BlockReduce<MinMax,256>(temp).Reduce(v,CombineBounds{});
    if(threadIdx.x==0){if(m.fixed_quantization){b.lo=m.quant_low;b.hi={m.quant_low.x+m.quant_extent.x,m.quant_low.y+m.quant_extent.y,m.quant_low.z+m.quant_extent.z};}*out={};out->low[0]=b.lo.x;out->low[1]=b.lo.y;out->low[2]=b.lo.z;out->extent[0]=m.fixed_quantization?m.quant_extent.x:b.hi.x-b.lo.x;out->extent[1]=m.fixed_quantization?m.quant_extent.y:b.hi.y-b.lo.y;out->extent[2]=m.fixed_quantization?m.quant_extent.z:b.hi.z-b.lo.z;out->invalid=b.invalid;
        for(unsigned j=0;j<3;++j)if(!isfinite(out->low[j])||!isfinite(out->extent[j])||out->extent[j]<0||!isfinite(out->low[j]+out->extent[j]))out->invalid|=DrawDomain;
        if(l.exact_bits){auto* bits=reinterpret_cast<uint32_t*>(buffer+l.exact_bits);auto* rank=reinterpret_cast<uint32_t*>(buffer+l.exact_rank);auto* used=reinterpret_cast<uint32_t*>(buffer+l.used);uint32_t count=0,referenced=0;
            for(uint32_t word=0;word<(m.vertices+31)/32;++word){rank[word]=count;count+=__popc(bits[word]);referenced+=__popc(used[word]);}
            if(uint64_t(count)*10000>uint64_t(referenced)*m.exact_position_bps)out->invalid|=DrawPrecisionCap;
            out->bits_offset=uint32_t(l.exact_bits/4);out->rank_offset=uint32_t(l.exact_rank/4);out->exact_offset=uint32_t(l.exact_positions/4);}
    }
}
__global__ void draw_streams(DeviceMeshView m,NeuralVertexStorage storage,DrawLayout l,char* buffer){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;auto& b=*reinterpret_cast<DrawMetadata*>(buffer+l.metadata);uint32_t faces=b.invalid?0:m.trial_status?(m.trial_status->invalid?0:m.trial_status->faces):m.faces;
    if(i==0){auto* args=reinterpret_cast<uint32_t*>(buffer+l.indirect);args[0]=faces*3;args[1]=faces?1:0;args[2]=args[3]=args[4]=0;}
    if((m.trial_status&&m.trial_status->invalid)||b.invalid)return;
    if(i<m.vertices){auto p=m.positions[i];
        if(l.exact_bits){uint32_t word=reinterpret_cast<uint32_t*>(buffer+l.exact_bits)[i/32],bit=1u<<(i%32);if(word&bit){auto rank=reinterpret_cast<uint32_t*>(buffer+l.exact_rank)[i/32]+__popc(word&(bit-1));reinterpret_cast<Vec3*>(buffer+l.exact_positions)[rank]=p;}}
        if(storage==NeuralVertexStorage::Float32)reinterpret_cast<Vec3*>(buffer+l.position)[i]=p;
        else {auto* q=reinterpret_cast<uint16_t*>(buffer+l.position)+3*i;q[0]=pack_unorm16(p.x,b.low[0],b.extent[0]);q[1]=pack_unorm16(p.y,b.low[1],b.extent[1]);q[2]=pack_unorm16(p.z,b.low[2],b.extent[2]);}
        if(m.normals||i==0){auto n=m.normals?m.normals[i]:Vec3{};if(storage==NeuralVertexStorage::Packed)reinterpret_cast<uint32_t*>(buffer+l.normal)[i]=pack_direction(n);else {double length=sqrt(double(n.x)*n.x+double(n.y)*n.y+double(n.z)*n.z);if(length>0)n={float(n.x/length),float(n.y/length),float(n.z/length)};reinterpret_cast<Vec3*>(buffer+l.normal)[i]=n;}}
        if(m.colors||i==0)reinterpret_cast<ColorRGBA8*>(buffer+l.color)[i]=m.colors?m.colors[i]:ColorRGBA8{255,255,255,255};
        if(m.uv){auto t=m.uv[i];if(storage==NeuralVertexStorage::Packed){auto* q=reinterpret_cast<uint16_t*>(buffer+l.uv)+2*i;q[0]=pack_unorm16(t.x,-8,16);q[1]=pack_unorm16(t.y,-8,16);}else reinterpret_cast<Vec2*>(buffer+l.uv)[i]=t;}
        if(m.tangents){auto t=m.tangents[i];if(storage==NeuralVertexStorage::Packed)reinterpret_cast<uint32_t*>(buffer+l.tangent)[i]=pack_direction({t.x,t.y,t.z},t.w<0?-1:1);else reinterpret_cast<Vec4*>(buffer+l.tangent)[i]=t;}
    }
    if(i<faces){auto* index=reinterpret_cast<uint32_t*>(buffer+l.indices);for(unsigned j=0;j<3;++j)index[3*i+j]=m.indices[3*i+j];uint32_t mat=m.materials?m.materials[i]:0;reinterpret_cast<uint32_t*>(buffer+l.faces)[i]=mat|((mat<m.sided_count&&m.double_sided[mat]?1u:0u)<<16);}
}
__global__ void referenced_positions(DeviceMeshView m,DrawLayout l,char* buffer){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,faces=m.trial_status?(m.trial_status->invalid?0:m.trial_status->faces):m.faces;if(i>=faces*3)return;auto v=m.indices[i],bit=1u<<(v%32);atomicOr(reinterpret_cast<uint32_t*>(buffer+l.used)+v/32,bit);if(m.exact_position_bits[v/32]&bit)atomicOr(reinterpret_cast<uint32_t*>(buffer+l.exact_bits)+v/32,bit);}
void pack_draw(DeviceMeshView m,NeuralVertexStorage storage,const DrawLayout& l,void* out,bool coverage_only){
    if(l.exact_bits){gpu::check(gpu::memset(static_cast<char*>(out)+l.exact_bits,0,((size_t(m.vertices)+31)/32)*4));gpu::check(gpu::memset(static_cast<char*>(out)+l.used,0,((size_t(m.vertices)+31)/32)*4));referenced_positions<<<gpu::blocks(size_t(m.faces)*3),256,0,gpu::stream()>>>(m,l,static_cast<char*>(out));}
    draw_bounds<<<1,256,0,gpu::stream()>>>(m,reinterpret_cast<DrawMetadata*>(static_cast<char*>(out)+l.metadata),storage==NeuralVertexStorage::Packed,l,static_cast<char*>(out));
    if(coverage_only){m.normals=nullptr;m.uv=nullptr;m.colors=nullptr;m.tangents=nullptr;}
    draw_streams<<<gpu::blocks(std::max(m.vertices,m.faces)),256,0,gpu::stream()>>>(m,storage,l,static_cast<char*>(out));gpu::check(cudaGetLastError());
}
__global__ void clip_draw(DeviceMeshView m,NeuralVertexStorage storage,DrawLayout l,char* buffer,Bounds bounds,Camera c,uint32_t size,uint8_t ss,uint32_t layer){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;uint32_t faces=m.trial_status?(m.trial_status->invalid?0:m.trial_status->faces):m.faces;if(i>=faces*3)return;auto& b=*reinterpret_cast<DrawMetadata*>(buffer+l.metadata);uint32_t vertex=m.indices[i];Vec3 p;
    if(b.invalid)return;
    if(storage==NeuralVertexStorage::Float32||(m.exact_position_bits&&((m.exact_position_bits[vertex/32]>>(vertex%32))&1)))p=m.positions[vertex];else{auto* q=reinterpret_cast<uint16_t*>(buffer+l.position)+3*vertex;p={unpack_unorm16(q[0],b.low[0],b.extent[0]),unpack_unorm16(q[1],b.low[1],b.extent[1]),unpack_unorm16(q[2],b.low[2],b.extent[2])};}
    double x=double(p.x)-bounds.center.x,y=double(p.y)-bounds.center.y,z=double(p.z)-bounds.center.z;
    double d=c.distance-(x*c.forward.x+y*c.forward.y+z*c.forward.z),scale=(c.perspective?c.focal/d:c.scale)*ss;
    double px=(x*c.right.x+y*c.right.y+z*c.right.z)*scale+size*.5,py=(x*c.up.x+y*c.up.y+z*c.up.z)*scale+size*.5;
    if(!isfinite(px)||!isfinite(py)||px<0||py<0||px>size||py>size||d<=fmax(bounds.radius*.01,c.distance-bounds.radius*2)||d>=c.distance+bounds.radius*2)atomicOr(&b.unused[0],1u<<layer);
}
__global__ void clear_draw_clip(DrawMetadata* metadata,uint32_t layer){metadata->unused[0]&=~(1u<<layer);}
void check_draw_clip(DeviceMeshView m,NeuralVertexStorage storage,const DrawLayout& l,void* out,const Bounds& b,const Camera& c,uint32_t size,uint8_t ss,uint32_t layer){
    if(layer>=8)throw std::invalid_argument("draw clip layer outside 0..7");
    clear_draw_clip<<<1,1,0,gpu::stream()>>>(reinterpret_cast<DrawMetadata*>(static_cast<char*>(out)+l.metadata),layer);
    clip_draw<<<gpu::blocks(size_t(m.faces)*3),256,0,gpu::stream()>>>(m,storage,l,static_cast<char*>(out),b,c,size,ss,layer);gpu::check(cudaGetLastError());
}
__global__ void gather_draw_status(DrawStatusSources s,uint32_t* out){uint32_t invalid=0,clipped=0;for(uint32_t i=0;i<s.count;++i){invalid|=uint32_t(s.metadata[i]->invalid!=0)<<i;clipped|=uint32_t((s.metadata[i]->unused[0]&(1u<<s.layers[i]))!=0)<<i;out[2+i]=s.trial[i]?(s.trial[i]->invalid?0:s.trial[i]->faces):s.faces[i];}out[0]=invalid;out[1]=clipped;}
void collect_draw_status(DrawStatusSources s,uint32_t* out){gather_draw_status<<<1,1,0,gpu::stream()>>>(s,out);gpu::check(cudaGetLastError());}
__global__ void unpack_targets(cudaSurfaceObject_t mask,cudaSurfaceObject_t attributes,cudaSurfaceObject_t rgb,AuditPixel* out,Vec3* color,uint32_t size,cudaSurfaceObject_t debug,RasterDebugPixel* witnesses){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size*size)return;auto x=i%size,y=i/size;
    auto a=attributes?surf2Dread<float4>(attributes,x*sizeof(float4),y):float4{};AuditPixel p{};p.normal={a.x,a.y,a.z};p.covered=surf2Dread<unsigned char>(mask,x,y)!=0;p.visible=a.w>0;p.material=p.visible?uint16_t(uint32_t(a.w)-1):0;out[i]=p;
    if(color){auto c=surf2Dread<float4>(rgb,x*sizeof(float4),y);color[i]={c.x,c.y,c.z};}
    if(witnesses){auto q=surf2Dread<uint2>(debug,x*sizeof(uint2),y);witnesses[i]={q.x?q.x-1:UINT32_MAX,q.x?__uint_as_float(q.y):INFINITY};}
}
__global__ void coverage_bits(cudaSurfaceObject_t mask,uint32_t* bits,uint32_t size){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    bool covered=i<size*size&&surf2Dread<unsigned char>(mask,i%size,i/size)!=0;
    uint32_t word=__ballot_sync(0xffffffffu,covered);
    if((threadIdx.x&31)==0&&i<size*size)bits[i/32]=word;
}
void pack_coverage_mask(cudaSurfaceObject_t mask,uint32_t* bits,uint32_t size){coverage_bits<<<gpu::blocks(size_t(size)*size),256,0,gpu::stream()>>>(mask,bits,size);gpu::check(cudaGetLastError());}
void unpack_hardware(cudaSurfaceObject_t mask,cudaSurfaceObject_t attributes,cudaSurfaceObject_t colors,AuditPixel* out,Vec3* rgb,uint32_t size,cudaSurfaceObject_t debug,RasterDebugPixel* witnesses){unpack_targets<<<gpu::blocks(size_t(size)*size),256,0,gpu::stream()>>>(mask,attributes,colors,out,rgb,size,debug,witnesses);gpu::check(cudaGetLastError());}
}
