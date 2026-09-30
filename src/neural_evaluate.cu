#include "neural_cuda.cuh"
#include "neural_audit_cache.hpp"
#include "metric_angle.hpp"
#include "neural_raster_pixel.cuh"
#include <cub/cub.cuh>
#include <math_constants.h>
namespace blitz::neural {
using namespace gpu;
namespace {
constexpr double infinity=std::numeric_limits<double>::infinity();
__device__ Vec3 add(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ Vec3 sub(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ Vec3 mul(Vec3 a,double b){return {float(a.x*b),float(a.y*b),float(a.z*b)};}
__device__ double dp(Vec3 a,Vec3 b){return double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z;}
__device__ Vec3 cp(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ double len(Vec3 a){return sqrt(dp(a,a));}
__device__ Vec3 norm(Vec3 a){double l=len(a);return l>0?mul(a,1/l):Vec3{};}
__device__ double edge(double ax,double ay,double bx,double by,double x,double y){return (bx-ax)*(y-ay)-(by-ay)*(x-ax);}
struct Vertex {double x,y,z;Vec3 normal;Vec4 color;};
struct PositionInput {
    const Vec3* full{};const uint16_t* packed{};Vec3 low{},step{};
    __device__ Vec3 operator[](uint32_t i)const{return packed?Vec3{low.x+step.x*packed[i*3],low.y+step.y*packed[i*3+1],low.z+step.z*packed[i*3+2]}:full[i];}
};
struct Triangle {Vertex a,b,c;Vec3 normal;double area,r0,r1,r2;int x0,x1,y0,y1;uint16_t material;uint8_t back,valid;};
struct Bins {unsigned long long entries;int clipped;};
struct Summary {unsigned long long ca,cb,changed,total,samples,pixels,normal;double distance,attribute;};
struct AddSummary {__device__ Summary operator()(Summary a,Summary b) const {return {a.ca+b.ca,a.cb+b.cb,a.changed+b.changed,a.total+b.total,a.samples+b.samples,a.pixels+b.pixels,max(a.normal,b.normal),fmax(a.distance,b.distance),fmax(a.attribute,b.attribute)};}};
__device__ void max_double(double* p,double x){atomicMax(reinterpret_cast<unsigned long long*>(p),__double_as_longlong(x));}
__global__ void project(PositionInput position,const Vec3* normals,const ColorRGBA8* colors,Vertex* out,uint32_t n,Bounds bounds,Camera camera,uint32_t size,uint8_t ss) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;auto a=position[i];double x=double(a.x)-bounds.center.x,y=double(a.y)-bounds.center.y,z0=double(a.z)-bounds.center.z;
    double z=camera.distance-(x*camera.forward.x+y*camera.forward.y+z0*camera.forward.z),scale=camera.perspective?camera.focal/z:camera.scale;
    Vertex p;p.x=(x*camera.right.x+y*camera.right.y+z0*camera.right.z)*scale*ss+size*.5;p.y=(x*camera.up.x+y*camera.up.y+z0*camera.up.z)*scale*ss+size*.5;p.z=z;
    p.normal=norm(normals?normals[i]:Vec3{});p.color=colors?Vec4{colors[i].r/255.f,colors[i].g/255.f,colors[i].b/255.f,colors[i].a/255.f}:Vec4{1,1,1,1};out[i]=p;
}
__global__ void triangles(const Vertex* verts,PositionInput pos,const uint32_t* indices,const uint16_t* materials,const uint8_t* two,uint32_t ntwo,
    Triangle* out,uint32_t* counts,Bins* bins,uint32_t nf,uint32_t size,double diameter,bool perspective,bool force_two) {
    uint32_t f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;counts[f]=0;Triangle t{};
    auto ia=indices[f*3],ib=indices[f*3+1],ic=indices[f*3+2];t.a=verts[ia];t.b=verts[ib];t.c=verts[ic];t.material=materials?materials[f]:0;
    if(perspective&&(t.a.z<=0||t.b.z<=0||t.c.z<=0)){atomicExch(&bins->clipped,1);out[f]=t;return;}
    t.area=edge(t.a.x,t.a.y,t.b.x,t.b.y,t.c.x,t.c.y);t.back=t.area<0;
    if(fabs(t.area)<1e-16||(t.back&&!force_two&&!(t.material<ntwo&&two[t.material]))){out[f]=t;return;}
    auto p0=pos[ia],p1=pos[ib],p2=pos[ic];Vec3 d1{float((double(p1.x)-p0.x)/diameter),float((double(p1.y)-p0.y)/diameter),float((double(p1.z)-p0.z)/diameter)};
    Vec3 d2{float((double(p2.x)-p0.x)/diameter),float((double(p2.y)-p0.y)/diameter),float((double(p2.z)-p0.z)/diameter)};t.normal=norm(cp(d1,d2));
    if(t.back){auto b=t.b;t.b=t.c;t.c=b;t.area=-t.area;t.normal=mul(t.normal,-1);}
    double minx=fmin(t.a.x,fmin(t.b.x,t.c.x)),maxx=fmax(t.a.x,fmax(t.b.x,t.c.x));
    double miny=fmin(t.a.y,fmin(t.b.y,t.c.y)),maxy=fmax(t.a.y,fmax(t.b.y,t.c.y));
    if(!isfinite(minx)||!isfinite(maxx)||!isfinite(miny)||!isfinite(maxy)||minx<0||miny<0||maxx>size||maxy>size){atomicExch(&bins->clipped,1);out[f]=t;return;}
    t.x0=max(0,int(floor(minx)));t.x1=min(int(size)-1,int(floor(maxx)));t.y0=max(0,int(floor(miny)));t.y1=min(int(size)-1,int(floor(maxy)));
    t.r0=.5*(fabs(t.c.x-t.b.x)+fabs(t.c.y-t.b.y));t.r1=.5*(fabs(t.a.x-t.c.x)+fabs(t.a.y-t.c.y));t.r2=.5*(fabs(t.b.x-t.a.x)+fabs(t.b.y-t.a.y));t.valid=1;
    counts[f]=uint32_t((t.x1/16-t.x0/16+1)*(t.y1/16-t.y0/16+1));atomicAdd(&bins->entries,static_cast<unsigned long long>(counts[f]));out[f]=t;
}
__global__ void scatter(const Triangle* tris,const uint32_t* offsets,uint64_t* keys,uint32_t n,uint32_t tiles) {
    uint32_t f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=n||!tris[f].valid)return;auto t=tris[f];uint32_t at=offsets[f];
    for(int y=t.y0/16;y<=t.y1/16;++y)for(int x=t.x0/16;x<=t.x1/16;++x)keys[at++]=(uint64_t(y*tiles+x)<<32)|f;
}
__global__ void ranges(const uint64_t* keys,uint32_t* begin,uint32_t* end,uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;auto tile=uint32_t(keys[i]>>32);
    if(!i||uint32_t(keys[i-1]>>32)!=tile)begin[tile]=i;if(i+1==n||uint32_t(keys[i+1]>>32)!=tile)end[tile]=i+1;
}
__device__ bool top_left(const Vertex& a,const Vertex& b){return b.y<a.y||(b.y==a.y&&b.x<a.x);}
__device__ bool same_vec(Vec3 a,Vec3 b){return __float_as_uint(a.x)==__float_as_uint(b.x)&&__float_as_uint(a.y)==__float_as_uint(b.y)&&__float_as_uint(a.z)==__float_as_uint(b.z);}
__device__ void dirty_box(uint32_t* dirty,uint32_t tiles,int x0,int x1,int y0,int y1,uint32_t size){
    x0=max(0,x0);y0=max(0,y0);x1=min(int(size)-1,x1);y1=min(int(size)-1,y1);
    for(int y=y0/16;y0<=y1&&y<=y1/16;++y)for(int x=x0/16;x0<=x1&&x<=x1/16;++x)atomicExch(dirty+y*tiles+x,1u);
}
// Face compaction preserves relative primitive order. A tile is reusable only
// when no removed or changed primitive touched it in either mesh. Vertex IDs,
// position/normal bits, RGB, material and both old/new bounds participate.
__global__ void dirty_tiles(DeviceMeshView before,DeviceMeshView after,const uint32_t* keep,const uint32_t* offsets,
    const Triangle* tris,uint32_t* dirty,uint32_t tiles,uint32_t size,Bounds bounds,Camera camera,uint8_t ss){
    uint32_t f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=before.faces)return;
    uint32_t next=offsets[f];bool changed=!keep[f];
    if(!changed){changed=(before.materials?before.materials[f]:0)!=(after.materials?after.materials[next]:0);
        for(unsigned j=0;j<3&&!changed;++j){auto a=before.indices[f*3+j],b=after.indices[next*3+j];
            changed=a!=b||!same_vec(before.positions[a],after.positions[b]);
            if(!changed&&before.normals&&after.normals)changed=!same_vec(before.normals[a],after.normals[b]);
            if(!changed&&before.colors&&after.colors){auto x=before.colors[a],y=after.colors[b];changed=x.r!=y.r||x.g!=y.g||x.b!=y.b;}
        }}
    if(!changed)return;
    double minx=CUDART_INF,miny=CUDART_INF,maxx=-CUDART_INF,maxy=-CUDART_INF;
    for(unsigned j=0;j<3;++j){auto p=before.positions[before.indices[f*3+j]];
        double x=double(p.x)-bounds.center.x,y=double(p.y)-bounds.center.y,z=double(p.z)-bounds.center.z;
        double depth=camera.distance-(x*camera.forward.x+y*camera.forward.y+z*camera.forward.z),scale=camera.perspective?camera.focal/depth:camera.scale;
        double px=(x*camera.right.x+y*camera.right.y+z*camera.right.z)*scale*ss+size*.5;
        double py=(x*camera.up.x+y*camera.up.y+z*camera.up.z)*scale*ss+size*.5;
        if(!isfinite(px)||!isfinite(py)||(camera.perspective&&depth<=0)){minx=miny=0;maxx=maxy=size;break;}
        minx=fmin(minx,px);maxx=fmax(maxx,px);miny=fmin(miny,py);maxy=fmax(maxy,py);
    }
    dirty_box(dirty,tiles,int(floor(fmax(0.,fmin(double(size),minx)))),int(floor(fmax(0.,fmin(double(size),maxx)))),
        int(floor(fmax(0.,fmin(double(size),miny)))),int(floor(fmax(0.,fmin(double(size),maxy)))),size);
    if(keep[f]&&tris[next].valid){const auto& t=tris[next];dirty_box(dirty,tiles,t.x0,t.x1,t.y0,t.y1,size);}
}
template<class Target,int DepthBits=32> __global__ void raster(const Triangle* tris,const uint64_t* keys,const uint32_t* begin,const uint32_t* end,Target* pixels,uint32_t size,
    uint32_t tiles,bool perspective,double diameter,Summary* summary,const Target* parent,const uint32_t* dirty,double depth_min,double depth_span,Vec3* colors,const Vec3* parent_colors,bool has_colors) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(size)*size)return;int x=int(i%size),y=int(i/size);auto tile=uint32_t(y/16)*tiles+x/16;
    if(parent&&!dirty[tile]){pixels[i]=parent[i];if(colors)colors[i]=parent_colors[i];return;}
    Pixel p{};p.depth=CUDART_INF_F;unsigned long long overlap=0;uint16_t nearest=65535;
    for(uint32_t k=begin[tile];k<end[tile];++k) {
        const auto& t=tris[uint32_t(keys[k])];if(x<t.x0||x>t.x1||y<t.y0||y>t.y1)continue;
        double u=edge(t.b.x,t.b.y,t.c.x,t.c.y,x+.5,y+.5),v=edge(t.c.x,t.c.y,t.a.x,t.a.y,x+.5,y+.5),w=edge(t.a.x,t.a.y,t.b.x,t.b.y,x+.5,y+.5);
        if(u+t.r0<0||v+t.r1<0||w+t.r2<0)continue;p.covered=1;
        if(u<0||v<0||w<0)continue;
        if((u>0||top_left(t.b,t.c))&&(v>0||top_left(t.c,t.a))&&(w>0||top_left(t.a,t.b)))++overlap;
        u/=t.area;v/=t.area;w/=t.area;double depth;
        if(perspective){double sum=u/t.a.z+v/t.b.z+w/t.c.z;depth=1/sum;u=u/t.a.z/sum;v=v/t.b.z/sum;w=w/t.c.z/sum;}
        else depth=u*t.a.z+v*t.b.z+w*t.c.z;depth/=diameter;
        // Process sorted source face IDs. This matches the reference's double comparison
        // against its previous float depth, including rounding-dependent ties.
        if constexpr(DepthBits==16){auto encoded=uint16_t(fmin(65535.,fmax(0.,floor((depth-depth_min)/depth_span*65535.+.5))));
            if(p.visible&&encoded>=nearest)continue;nearest=encoded;p.depth=float(depth_min+double(encoded)/65535.*depth_span);}
        else {if(depth>=p.depth)continue;p.depth=float(depth);}p.visible=1;p.material=t.material;
        p.normal=norm(add(add(mul(t.a.normal,u),mul(t.b.normal,v)),mul(t.c.normal,w)));
        if(len(p.normal)<.5)p.normal=t.normal;else if(t.back)p.normal=mul(p.normal,-1);
        p.color=has_colors?Vec4{float(u*t.a.color.x+v*t.b.color.x+w*t.c.color.x),float(u*t.a.color.y+v*t.b.color.y+w*t.c.color.y),float(u*t.a.color.z+v*t.b.color.z+w*t.c.color.z),1}:Vec4{1,1,1,1};
    }
    pixels[i]=Target(p);if(colors)colors[i]={p.color.x,p.color.y,p.color.z};if(summary&&overlap){atomicAdd(&summary->samples,overlap);atomicAdd(&summary->pixels,1ull);}
}
template<class T> struct ImageOf {Buffer<T> pixels;Buffer<Vec3> colors;uint32_t size;bool clipped;};
using Image=ImageOf<AuditPixel>;
// Exact owned mesh keys also cover mutable positions/normals. A topology-only
// key is insufficient as soon as a proposal changes shading or placement.
class RasterMemo {
    using Key=std::tuple<float,float,float,double,double,uint32_t,uint16_t,uint16_t,bool>;
    struct Entry {uint32_t view;uint8_t sampling;std::unique_ptr<Image> image;};
    Key key_{};Mesh mesh_;DeviceMeshView device_mesh_{};size_t limit_,bytes_{};std::vector<Entry> entries_;bool enabled_{true};
    static Key key(const Bounds& b,const EvalSettings& e){return {b.center.x,b.center.y,b.center.z,b.radius,e.screen_size,e.views.rotation_seed,e.views.orthographic,e.views.perspective,e.force_two_sided};}
public:
    explicit RasterMemo(size_t limit):limit_(limit){}
    void clear(){entries_.clear();bytes_=0;}
    void configure(MeshView m,const Bounds& b,const EvalSettings& e){
        auto next=key(b,e);
        try{if(device_mesh_.identity||next!=key_||mesh_.positions.empty()||!same_mesh_data(m,mesh_.view())){clear();mesh_=copy_mesh(m);key_=next;device_mesh_={};}enabled_=true;}
        catch(const std::bad_alloc&){clear();enabled_=false;}}
    void configure(DeviceMeshView m,const Bounds& b,const EvalSettings& e){auto next=key(b,e);
        if(!m.identity)throw std::invalid_argument("device render cache requires owned mesh identity");
        auto& old=device_mesh_;
        if(next!=key_||old.identity!=m.identity||old.revision!=m.revision||old.positions!=m.positions||old.normals!=m.normals||old.colors!=m.colors||old.indices!=m.indices||old.materials!=m.materials||old.double_sided!=m.double_sided||old.vertices!=m.vertices||old.faces!=m.faces||old.sided_count!=m.sided_count){clear();device_mesh_=m;key_=next;mesh_={};}enabled_=true;}
    const Image* find(uint32_t view,uint8_t sampling)const{if(enabled_)for(auto& entry:entries_)if(entry.view==view&&entry.sampling==sampling)return entry.image.get();return nullptr;}
    bool room(size_t bytes)const{return enabled_&&bytes<=limit_-bytes_&&entries_.size()<4096;}
    const Image* retain(uint32_t view,uint8_t sampling,Image&& image){size_t bytes=image.pixels.n*sizeof(AuditPixel)+image.colors.n*sizeof(Vec3);
        if(!enabled_||entries_.size()>=4096||bytes>limit_-bytes_)return nullptr;
        // Reserve metadata before moving the only owned raster. Allocation
        // failure leaves the caller's scratch image usable.
        std::unique_ptr<Image> saved;
        try{entries_.reserve(entries_.size()+1);saved=std::make_unique<Image>(std::move(image));}catch(const std::bad_alloc&){return nullptr;}
        auto* result=saved.get();entries_.push_back({view,sampling,std::move(saved)});bytes_+=bytes;return result;}
};
struct UploadedMesh {
    Buffer<Vec3> positions,normals;Buffer<ColorRGBA8> colors;
    Buffer<uint32_t> indices;Buffer<uint16_t> materials;Buffer<uint8_t> double_sided;
    const Vec3 *position,*normal;const ColorRGBA8* color;const uint8_t* sided;
    uint32_t vertices,faces,sided_count;
    UploadedMesh(Device& d,MeshView m,const UploadedMesh* shared=nullptr,bool upload_positions=true):
        positions(shared||!upload_positions?Buffer<Vec3>{}:upload_stream(d,m.positions)),normals(shared?Buffer<Vec3>{}:upload_stream(d,m.normals)),
        colors(shared?Buffer<ColorRGBA8>{}:upload_stream(d,m.colors)),indices(d,m.indices.size()),materials(d,m.materials.size()),double_sided(d,shared?0:m.double_sided.size()),
        position(shared?shared->position:positions.p),normal(shared?shared->normal:normals.p),color(shared?shared->color:colors.p),sided(shared?shared->sided:double_sided.p),
        vertices(uint32_t(m.positions.count)),faces(uint32_t(m.triangles())),sided_count(uint32_t(m.double_sided.size())) {
        indices.upload(m.indices);materials.upload(m.materials);if(!shared)double_sided.upload(m.double_sided);
    }
    operator DeviceMeshView()const{return {position,normal,nullptr,color,nullptr,indices.p,materials.p,sided,vertices,faces,sided_count,0,0};}
};
template<class Target=AuditPixel,int DepthBits=32> ImageOf<Target> render(Device& device,DeviceMeshView m,const Bounds& b,const Camera& camera,double screen,uint8_t ss,bool two,Summary* summary=nullptr,const ImageOf<Target>* parent=nullptr,double depth_min=0,double depth_span=1,PositionInput positions={}) {
    if(!ss||!std::isfinite(screen)||screen<=0||screen>16384)throw std::invalid_argument("invalid raster extent");
    uint32_t size=summary?(uint32_t(std::ceil(screen+8))+1)&~1u:uint32_t(std::ceil(screen+8))*ss;
    if(uint64_t(size)*size>max_raster_samples)throw ResourceError(NeuralResourceLimit::SampleCount,uint64_t(size)*size,max_raster_samples,"raster exceeds per-view sample cap");
    Buffer<Vertex> projected(device,m.vertices);uint32_t nf=m.faces;
    Buffer<Triangle> tris(device,nf);Buffer<uint32_t> counts(device,nf+1),offsets(device,nf+1);counts.zero();Buffer<Bins> bins(device,1);bins.zero();
    if(!positions.packed)positions.full=m.positions;
    project<<<blocks(m.vertices),256>>>(positions,m.normals,m.colors,projected.p,m.vertices,b,camera,size,ss);
    triangles<<<blocks(nf),256>>>(projected.p,positions,m.indices,m.materials,m.double_sided,m.sided_count,tris.p,counts.p,bins.p,nf,size,b.diameter(),camera.perspective,two);check(cudaGetLastError());
    // Check the exact 64-bit sum before a 32-bit scan. The clipping flag is
    // already available here, so it shares the same host synchronization.
    auto bin_result=bins.download()[0];uint64_t entries=bin_result.entries;
    if(entries>INT32_MAX)throw ResourceError(NeuralResourceLimit::TileEntries,entries,INT32_MAX,"CUDA tile list exceeds 31-bit sort bound");
    size_t bytes=0;check(cub::DeviceScan::ExclusiveSum(nullptr,bytes,counts.p,offsets.p,nf+1));
    {Buffer<std::byte> temp(device,bytes);check(cub::DeviceScan::ExclusiveSum(temp.p,bytes,counts.p,offsets.p,nf+1));}
    uint32_t tile_width=(size+15)/16,ntiles=tile_width*tile_width;
    Buffer<uint64_t> keys(device,size_t(entries)),sorted(device,size_t(entries));Buffer<uint32_t> begin(device,ntiles),end(device,ntiles);begin.zero();end.zero();
    if(entries) {
        scatter<<<blocks(nf),256>>>(tris.p,offsets.p,keys.p,nf,tile_width);check(cudaGetLastError());
        check(cub::DeviceRadixSort::SortKeys(nullptr,bytes,keys.p,sorted.p,int(entries)));
        {Buffer<std::byte> temp(device,bytes);check(cub::DeviceRadixSort::SortKeys(temp.p,bytes,keys.p,sorted.p,int(entries)));}
        ranges<<<blocks(entries),256>>>(sorted.p,begin.p,end.p,uint32_t(entries));check(cudaGetLastError());
    }
    Buffer<Target> pixels(device,size_t(size)*size);
    Buffer<Vec3> colors(device,std::is_same_v<Target,AuditPixel>&&m.colors?pixels.n:0);
    Buffer<uint32_t> dirty(device,parent?ntiles:0);
    if(parent){dirty.zero();dirty_tiles<<<blocks(m.raster_parent->faces),256>>>(*m.raster_parent,m,m.parent_keep,m.parent_offsets,tris.p,dirty.p,tile_width,size,b,camera,ss);check(cudaGetLastError());}
    raster<Target,DepthBits><<<blocks(pixels.n),256>>>(tris.p,sorted.p,begin.p,end.p,pixels.p,size,tile_width,camera.perspective,b.diameter(),summary,parent?parent->pixels.p:nullptr,dirty.p,depth_min,depth_span,colors.p,parent?parent->colors.p:nullptr,m.colors!=nullptr);check(cudaGetLastError());
    return {std::move(pixels),std::move(colors),size,bin_result.clipped!=0};
}
__global__ void initialize(const AuditPixel* a,const AuditPixel* b,float* da,float* db,Summary* s,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;Summary local{};
    if(i<n){bool ca=a[i].covered,cb=b[i].covered;da[i]=ca?0.f:1e15f;db[i]=cb?0.f:1e15f;local.ca=ca;local.cb=cb;local.changed=ca!=cb;local.total=ca||cb;
        if(a[i].visible&&b[i].visible){auto bits=__double_as_longlong(fmin(1.,fmax(-1.,dp(a[i].normal,b[i].normal))));
            local.normal=(bits&0x8000000000000000ull)?bits:~(bits^0x8000000000000000ull);}}
    __shared__ cub::BlockReduce<Summary,256>::TempStorage storage;auto reduced=cub::BlockReduce<Summary,256>(storage).Reduce(local,AddSummary{});
    if(threadIdx.x==0){atomicAdd(&s->ca,reduced.ca);atomicAdd(&s->cb,reduced.cb);atomicAdd(&s->changed,reduced.changed);atomicAdd(&s->total,reduced.total);atomicMax(&s->normal,reduced.normal);}
}
// Transposing lets adjacent lanes visit adjacent pixels in both separable
// passes. The padded shared tile also avoids bank conflicts on the write pass.
__device__ bool needs_distance(const Summary* s){return s->changed&&bool(s->ca)==bool(s->cb);}
__global__ void transpose(const float* in,float* out,int size,const Summary* summary) {
    if(!needs_distance(summary))return;
    __shared__ float tile[32][33];
    int x=int(blockIdx.x)*32+int(threadIdx.x),y=int(blockIdx.y)*32+int(threadIdx.y);
    for(int j=0;j<32;j+=8)if(x<size&&y+j<size)tile[threadIdx.y+j][threadIdx.x]=in[size_t(y+j)*size+x];
    __syncthreads();
    x=int(blockIdx.y)*32+int(threadIdx.x);y=int(blockIdx.x)*32+int(threadIdx.y);
    for(int j=0;j<32;j+=8)if(x<size&&y+j<size)out[size_t(y+j)*size+x]=tile[threadIdx.x][threadIdx.y+j];
}
// The first pass sees only zero/1e15f coverage values. The nearest foreground
// site on either side therefore gives the same squared distance as the general
// lower envelope. At the raster cap, all finite distances are far below 1e15f;
// an empty column remains exactly that sentinel. Square before rounding to float.
__global__ void edt_binary(const float* in,float* out,int size,const Summary* summary) {
    if(!needs_distance(summary))return;
    int line=int(blockIdx.x*blockDim.x+threadIdx.x);if(line>=size)return;
    int nearest=-1;
    for(int q=0;q<size;++q){size_t i=size_t(q)*size+line;if(in[i]==0)nearest=q;double d=q-nearest;out[i]=nearest<0?1e15f:float(d*d);}
    nearest=size;
    for(int q=size-1;q>=0;--q){size_t i=size_t(q)*size+line;if(in[i]==0)nearest=q;double d=nearest-q;if(nearest<size)out[i]=fminf(out[i],float(d*d));}
}
// Exact separable squared Euclidean distance transform. One independent column
// per thread. Stack entries are [depth][column], so nearby lanes share memory
// transactions. Arithmetic and float rounding match the original row/column
// implementation; the two transposes change only the memory layout.
__global__ void edt(const float* in,float* out,int* stack,double* boundaries,int size,const Summary* summary) {
    if(!needs_distance(summary))return;
    int line=int(blockIdx.x*blockDim.x+threadIdx.x);if(line>=size)return;
    int* v=stack+line;double* z=boundaries+line;int k=0;v[0]=0;z[0]=-CUDART_INF;z[size]=CUDART_INF;
    for(int q=1;q<size;++q){double s;for(;;){int p=v[size_t(k)*size];s=((double(in[size_t(q)*size+line])+double(q)*q)-(double(in[size_t(p)*size+line])+double(p)*p))/(2*(q-p));if(s>z[size_t(k)*size]||k==0)break;--k;}++k;v[size_t(k)*size]=q;z[size_t(k)*size]=s;z[size_t(k+1)*size]=CUDART_INF;}
    k=0;for(int q=0;q<size;++q){while(z[size_t(k+1)*size]<q)++k;int p=v[size_t(k)*size];double d=q-p;out[size_t(q)*size+line]=float(d*d+in[size_t(p)*size+line]);}
}
__global__ void directed_coverage(const AuditPixel* from,const float* to,Summary* summary,size_t n) {
    if(!needs_distance(summary))return;
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double value=i<n&&from[i].covered?double(to[i]):0;
    __shared__ cub::BlockReduce<double,256>::TempStorage storage;double maximum=cub::BlockReduce<double,256>(storage).Reduce(value,cub::Max());
    if(threadIdx.x==0)max_double(&summary->distance,maximum);
}
struct SamplePixel {Vec3 normal,color;uint16_t material;uint8_t covered,visible;};
__device__ SamplePixel sample_pixel(const AuditPixel* pixels,const Vec3* colors,size_t i){auto p=pixels[i];return {p.normal,colors?colors[i]:Vec3{1,1,1},p.material,p.covered,p.visible};}
__device__ double sample(const SamplePixel& a,const SamplePixel& b,double spatial,int profile,Weights weights) {
    double cost=spatial;
    if(profile&&weights.normal>0){double cosine=fmin(1.,fmax(-1.,dp(a.normal,b.normal)/fmax(1e-30,len(a.normal)*len(b.normal))));double angle=detail::metric_acos(cosine)*weights.normal;cost+=angle*angle;}
    if(profile==2){double x=double(a.color.x)-b.color.x,y=double(a.color.y)-b.color.y,z=double(a.color.z)-b.color.z;cost+=weights.color*weights.color*(x*x+y*y+z*z);if(a.material!=b.material)cost+=weights.material*weights.material;}return cost;
}
__device__ float square_up(float x){return __fmul_ru(x,x);}
__device__ float difference_up(float a,float b){return __fsub_ru(fmaxf(a,b),fminf(a,b));}
__device__ float distance_up(Vec3 a,Vec3 b){return __fadd_ru(__fadd_ru(square_up(difference_up(a.x,b.x)),square_up(difference_up(a.y,b.y))),square_up(difference_up(a.z,b.z)));}
__device__ float length_down(Vec3 a){return __fadd_rd(__fadd_rd(__fmul_rd(a.x,a.x),__fmul_rd(a.y,a.y)),__fmul_rd(a.z,a.z));}
// This is an upper bound, not a replacement metric. For lengths >= 1/2,
// angle^2 <= pi^2*|a-b|^2/min(|a|^2,|b|^2) <= 4*pi^2*|a-b|^2.
// 40 > 4*pi^2. The 2^-40 guard exceeds 64*pi^2*DBL_EPSILON,
// covering FP64 dot/normalization/acos roundoff (including identical vectors).
// Directed FP32 rounding and one final upward ULP also cover cost accumulation.
// Pixels above this conservative bound always use the original FP64 metric.
__device__ bool below_coverage(const SamplePixel& a,const SamplePixel& b,int profile,Weights weights,float bound){
    if(!b.visible)return false;float upper=0;
    if(profile&&weights.normal>0){if(!(length_down(a.normal)>=.25f&&length_down(b.normal)>=.25f))return false;
        float angle=__fadd_ru(__fmul_ru(40.f,distance_up(a.normal,b.normal)),0x1p-40f);
        upper=__fmul_ru(square_up(__double2float_ru(weights.normal)),angle);}
    if(profile==2){float color=distance_up({a.color.x,a.color.y,a.color.z},{b.color.x,b.color.y,b.color.z});
        upper=__fadd_ru(upper,__fmul_ru(square_up(__double2float_ru(weights.color)),color));
        if(a.material!=b.material)upper=__fadd_ru(upper,square_up(__double2float_ru(weights.material)));}
    return nextafterf(upper,CUDART_INF_F)<=bound;
}
__device__ double appearance_pixel(const AuditPixel* from,const AuditPixel* to,const Vec3* from_colors,const Vec3* to_colors,size_t i,int size,int ss,double limit,int profile,Weights weights) {
    if(i>=size_t(size)*size||!from[i].visible)return 0;
    if(profile==0||(!weights.normal&&(profile!=2||(!weights.color&&!weights.material))))return 0;
    auto p=sample_pixel(from,from_colors,i);double best=to[i].visible?sample(p,sample_pixel(to,to_colors,i),0,profile,weights):CUDART_INF;if(best<=1e-18)return 0;
    int x=int(i%size),y=int(i/size),radius=int(fmin(double(size),ceil(limit*ss)));double limit2=limit*limit,invs2=1./(ss*ss);
    // Any other sample has spatial cost at least invs2 and nonnegative attribute
    // cost. Keep the computed center cost (including its rounding), but avoid a
    // search that cannot improve it. The center itself was already evaluated.
    if(best<=invs2)return best>limit2?CUDART_INF:best;
    int r=int(fmin(double(radius),ceil(sqrt(fmin(best,limit2))*ss)));
    for(int yy=max(0,y-r);yy<=min(size-1,y+r);++yy)for(int xx=max(0,x-r);xx<=min(size-1,x+r);++xx){if(xx==x&&yy==y)continue;size_t other=size_t(yy)*size+xx;if(!to[other].visible)continue;double dx=x-xx,dy=y-yy,spatial=(dx*dx+dy*dy)*invs2;if(spatial>=best||spatial>limit2)continue;best=fmin(best,sample(p,sample_pixel(to,to_colors,other),spatial,profile,weights));}
    return best>limit2?CUDART_INF:best;
}
__global__ void appearance(const AuditPixel* from,const AuditPixel* to,const Vec3* from_colors,const Vec3* to_colors,Summary* summary,int size,int ss,double limit,int profile,Weights weights,bool clipped,float floor_squared) {
    if(clipped||bool(summary->ca)!=bool(summary->cb)||sqrt(summary->distance)/ss+2*sqrt(2.)/ss+1e-6>limit)return;
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    double value=0;if(i<size_t(size)*size&&from[i].visible&&!below_coverage(sample_pixel(from,from_colors,i),sample_pixel(to,to_colors,i),profile,weights,floor_squared))value=appearance_pixel(from,to,from_colors,to_colors,i,size,ss,limit,profile,weights);
    // Every lane participates, including invisible/out-of-image pixels. Preserve
    // exact nonnegative maxima while issuing only one global atomic per block.
    __shared__ cub::BlockReduce<double,256>::TempStorage storage;
    double maximum=cub::BlockReduce<double,256>(storage).Reduce(value,cub::Max());
    if(threadIdx.x==0&&maximum>0)max_double(&summary->attribute,maximum);
}
Measurement measure(Device& device,DeviceMeshView a,DeviceMeshView b,const Bounds& bounds,const Camera& camera,const EvalSettings& config,uint8_t ss,
    RasterMemo& reference,RasterMemo& candidate,RasterMemo& parents,uint32_t view,NeuralStats* stats) {
    std::optional<Image> reference_scratch,candidate_scratch;
    auto image=[&](DeviceMeshView m,RasterMemo& cache,std::optional<Image>& scratch,bool is_reference)->const Image&{
        if(auto* cached=cache.find(view,ss)){if(stats){if(is_reference)++stats->gpu_reference_render_hits;else ++stats->gpu_candidate_render_hits;}return *cached;}
        const Image* parent=nullptr;
        if(!is_reference&&m.raster_parent&&m.parent_keep&&m.parent_offsets){
            const auto& before=*m.raster_parent;
            // Trial owners preserve attribute stream presence and sidedness.
            if(bool(before.normals)==bool(m.normals)&&bool(before.colors)==bool(m.colors)&&before.double_sided==m.double_sided&&before.sided_count==m.sided_count){
                parents.configure(before,bounds,config);parent=parents.find(view,ss);
                size_t extent=size_t(std::ceil(config.screen_size+8))*ss;
                if(!parent&&parents.room(extent*extent*(sizeof(AuditPixel)+(m.colors?sizeof(Vec3):0)))){
                    try{auto base=render(device,before,bounds,camera,config.screen_size,ss,config.force_two_sided);if(stats)++stats->gpu_rasters;parent=parents.retain(view,ss,std::move(base));}
                    catch(const ResourceError& error){if(error.kind!=NeuralResourceLimit::WorkspaceMemory&&error.kind!=NeuralResourceLimit::DeviceMemory)throw;parents.clear();}
                }
            }
        }
        scratch.emplace(render(device,m,bounds,camera,config.screen_size,ss,config.force_two_sided,nullptr,parent));if(stats)++stats->gpu_rasters;
        if(auto* retained=cache.retain(view,ss,std::move(*scratch))){scratch.reset();return *retained;}return *scratch;};
    auto& x=image(a,reference,reference_scratch,true);auto& y=image(b,candidate,candidate_scratch,false);
    size_t n=x.pixels.n;int size=int(x.size);Buffer<float> da(device,n),db(device,n),temp(device,n);Buffer<int> stack(device,n);Buffer<double> boundaries(device,size_t(size)*(size+1));Buffer<Summary> summary(device,1);summary.zero();
    initialize<<<blocks(n),256>>>(x.pixels.p,y.pixels.p,da.p,db.p,summary.p,n);check(cudaGetLastError());
    {
        for(auto pair:{std::pair{da.p,y.pixels.p},std::pair{db.p,x.pixels.p}}) {
            dim3 tiles((size+31)/32,(size+31)/32),threads(32,8);
            transpose<<<tiles,threads>>>(pair.first,temp.p,size,summary.p);
            edt_binary<<<(size+63)/64,64>>>(temp.p,pair.first,size,summary.p);
            transpose<<<tiles,threads>>>(pair.first,temp.p,size,summary.p);
            edt<<<(size+63)/64,64>>>(temp.p,pair.first,stack.p,boundaries.p,size,summary.p);
            directed_coverage<<<blocks(n),256>>>(pair.second,pair.first,summary.p,n);check(cudaGetLastError());
        }
    }
    if(config.profile!=Profile::Coverage){
        double floor=2*std::sqrt(2.)/ss+1e-6,squared=std::nextafter(floor*floor,0.);float bound=float(squared);if(double(bound)>squared)bound=std::nextafter(bound,0.f);
        appearance<<<blocks(n),256>>>(x.pixels.p,y.pixels.p,x.colors.p,y.colors.p,summary.p,size,ss,config.limit,int(config.profile),config.weights,x.clipped||y.clipped,bound);
        appearance<<<blocks(n),256>>>(y.pixels.p,x.pixels.p,y.colors.p,x.colors.p,summary.p,size,ss,config.limit,int(config.profile),config.weights,x.clipped||y.clipped,bound);check(cudaGetLastError());}
    auto s=summary.download()[0];
    Measurement m;m.supersample=ss;m.coverage=bool(s.ca)!=bool(s.cb)?infinity:std::sqrt(s.distance)/ss;
    m.coverage_upper=m.coverage+2*std::sqrt(2.)/ss+1e-6;if(x.clipped||y.clipped)m.coverage_upper=infinity;
    m.changed_area=s.total?double(s.changed)/s.total:0;m.error=m.coverage_upper;
    if(config.profile!=Profile::Coverage&&s.normal){uint64_t bits=(s.normal&0x8000000000000000ull)?s.normal:~s.normal^0x8000000000000000ull;
        m.normal_degrees=detail::metric_acos(std::bit_cast<double>(bits))*180/3.14159265358979323846;}
    if(m.coverage_upper<=config.limit&&config.profile!=Profile::Coverage) {
        m.error=std::max(m.error,std::sqrt(s.attribute));
    }
    m.passed=m.error<=config.limit&&m.changed_area<=config.max_changed_area;return m;
}
}
Raster raster_cuda(MeshView m,const Bounds& b,const Camera& c,double screen,uint8_t ss,bool two,const NeuralOptions& options) {
    if(auto error=validate(m);!error.empty())throw std::invalid_argument(error);Device device(options,true);UploadedMesh uploaded(device,m);auto image=render<Pixel>(device,uploaded,b,c,screen,ss,two);return {image.size,image.size,image.pixels.download(),image.clipped};
}
RasterPrecisionResult raster_precision_cuda(MeshView m,const Bounds& b,const Camera& c,double screen,uint8_t ss,bool two,uint8_t attribute_bits,uint8_t depth_bits,uint8_t position_bits,const NeuralOptions& options){
    if(auto error=validate(m);!error.empty())throw std::invalid_argument(error);
    if((attribute_bits!=8&&attribute_bits!=16&&attribute_bits!=32)||(depth_bits!=16&&depth_bits!=32)||(position_bits!=16&&position_bits!=32))throw std::invalid_argument("unsupported raster precision probe");
    Device device(options,true);UploadedMesh uploaded(device,m,nullptr,position_bits==32);PositionInput positions;Buffer<uint16_t> packed(device,position_bits==16?m.positions.count*3:0);
    if(position_bits==16){Vec3 low=m.positions[0],high=low;for(size_t i=1;i<m.positions.count;++i){auto p=m.positions[i];low={std::min(low.x,p.x),std::min(low.y,p.y),std::min(low.z,p.z)};high={std::max(high.x,p.x),std::max(high.y,p.y),std::max(high.z,p.z)};}
        positions={nullptr,packed.p,low,{float((double(high.x)-low.x)/65535),float((double(high.y)-low.y)/65535),float((double(high.z)-low.z)/65535)}};
        std::vector<uint16_t> codes(m.positions.count*3);for(size_t i=0;i<m.positions.count;++i){auto p=m.positions[i];auto encode=[](float x,float lo,float hi){return hi>lo?uint16_t(std::clamp(std::round((double(x)-lo)/(double(hi)-lo)*65535),0.,65535.)):uint16_t(0);};codes[i*3]=encode(p.x,low.x,high.x);codes[i*3+1]=encode(p.y,low.y,high.y);codes[i*3+2]=encode(p.z,low.z,high.z);}packed.upload(codes);}
    check(cudaDeviceSynchronize());double lo=INFINITY,hi=-INFINITY;
    for(size_t i=0;i<m.positions.count;++i){auto p=m.positions[i];double x=double(p.x)-b.center.x,y=double(p.y)-b.center.y,z=double(p.z)-b.center.z;
        double depth=(c.distance-(x*c.forward.x+y*c.forward.y+z*c.forward.z))/b.diameter();lo=std::min(lo,depth);hi=std::max(hi,depth);}
    auto run=[&]<class T,int Depth>(){{auto warmup=render<T,Depth>(device,uploaded,b,c,screen,ss,two,nullptr,nullptr,lo,hi>lo?hi-lo:1.,positions);check(cudaDeviceSynchronize());}
        auto start=std::chrono::steady_clock::now();auto image=render<T,Depth>(device,uploaded,b,c,screen,ss,two,nullptr,nullptr,lo,hi>lo?hi-lo:1.,positions);check(cudaDeviceSynchronize());
        RasterPrecisionResult result;result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();result.bytes_per_pixel=sizeof(T)+(image.colors.n?sizeof(Vec3):0);
        result.raster.width=result.raster.height=image.size;result.raster.clipped=image.clipped;auto pixels=image.pixels.download();auto rgb=image.colors.download();result.raster.pixels.reserve(pixels.size());for(size_t i=0;i<pixels.size();++i){Pixel p=Pixel(pixels[i]);if(!rgb.empty())p.color={rgb[i].x,rgb[i].y,rgb[i].z,1};result.raster.pixels.push_back(p);}return result;};
    if(depth_bits==16){if(attribute_bits==8)return run.template operator()<Pixel8,16>();if(attribute_bits==16)return run.template operator()<Pixel16,16>();return run.template operator()<AuditPixel,16>();}
    if(attribute_bits==8)return run.template operator()<Pixel8,32>();if(attribute_bits==16)return run.template operator()<Pixel16,32>();return run.template operator()<AuditPixel,32>();
}
struct AuditCuda::Impl {
    struct Topology {
        std::vector<uint32_t> indices;std::vector<uint16_t> materials;Mesh owned;bool standalone{};std::unique_ptr<UploadedMesh> uploaded;
        bool matches(MeshView m) const {return uploaded&&std::equal(indices.begin(),indices.end(),m.indices.begin(),m.indices.end())&&std::equal(materials.begin(),materials.end(),m.materials.begin(),m.materials.end());}
        const UploadedMesh* get(Device& device,MeshView m,const UploadedMesh* source){
            bool same=uploaded&&standalone==!source&&(source?matches(m):same_mesh_data(m,owned.view()));
            if(!same){uploaded.reset();standalone=!source;if(source){owned={};indices.assign(m.indices.begin(),m.indices.end());materials.assign(m.materials.begin(),m.materials.end());}else {owned=copy_mesh(m);indices.clear();materials.clear();}uploaded=std::make_unique<UploadedMesh>(device,m,source);}
            return uploaded.get();
        }
    };
    Device device;int id;MeshView source;AuditMemo memo;std::unique_ptr<UploadedMesh> uploaded;
    RasterMemo source_images,reference_images,candidate_images,parent_images;
    Topology reference,candidate;
    void clear_images(){source_images.clear();reference_images.clear();candidate_images.clear();parent_images.clear();}
    Measurement run(DeviceMeshView a,DeviceMeshView b,const Bounds& bounds,const EvalSettings& config,RasterMemo& images,NeuralStats* stats,uint32_t& view,uint8_t& sampling,double incumbent=infinity,bool* pruned=nullptr){
        Measurement result;result.supersample=config.supersample;auto views=cameras(bounds,config.screen_size,config.views);
        for(uint32_t v=0;v<views.size();++v){view=v;
            if(config.cancelled&&config.cancelled()){result.complete=false;result.passed=false;return result;}
            Measurement current;
            for(unsigned ss=config.supersample;;ss=std::min<unsigned>(config.max_supersample,ss*2)){sampling=uint8_t(ss);
                auto perform=[&]{return measure(device,a,b,bounds,views[v],config,sampling,images,candidate_images,parent_images,v,stats);};
                try{current=perform();}catch(const ResourceError& error){
                    if(error.kind!=NeuralResourceLimit::WorkspaceMemory&&error.kind!=NeuralResourceLimit::DeviceMemory)throw;
                    clear_images();current=perform();}
                if(current.passed||ss>=config.max_supersample||current.coverage-2*std::sqrt(2.)/ss>config.limit||!std::isfinite(current.error)||current.coverage_upper<=config.limit)break;}
            ++result.views_evaluated;if(current.error>result.error){result.worst_view=v;result.error=current.error;result.supersample=current.supersample;}
            result.coverage=std::max(result.coverage,current.coverage);result.coverage_upper=std::max(result.coverage_upper,current.coverage_upper);result.normal_degrees=std::max(result.normal_degrees,current.normal_degrees);
            if(current.changed_area>result.changed_area){result.changed_area=current.changed_area;result.changed_area_worst_view=v;}
            // Refinement above uses the ORIGINAL visual limit. Tightening it
            // would change sampling and could change the winning candidate.
            if(pruned&&(current.error/config.limit>=incumbent||(config.max_changed_area>0&&current.changed_area/config.max_changed_area>=incumbent))){*pruned=true;result.passed=false;result.complete=false;return result;}
            if(!current.passed){result.passed=false;result.complete=false;return result;}}
        return result;
    }
    Impl(const NeuralOptions& options,MeshView mesh):device(options,true),id(options.device),source(mesh),memo(mesh),
        source_images(options.cache_rasters?device.limit/12:0),reference_images(options.cache_rasters?device.limit/12:0),candidate_images(options.cache_rasters?device.limit/12:0),parent_images(options.cache_rasters?device.limit/12:0){check(cudaSetDevice(device.previous));}
    ~Impl(){cudaSetDevice(id);}
};
struct CurrentDevice {int previous;explicit CurrentDevice(int id){check(cudaGetDevice(&previous));check(cudaSetDevice(id));}~CurrentDevice(){cudaSetDevice(previous);}};
struct AuditCounters {Device& d;NeuralStats* s;uint64_t allocations,reuses,upload,download;
    AuditCounters(Device& device,NeuralStats* stats):d(device),s(stats),allocations(d.allocations),reuses(d.reuses),upload(d.upload_bytes),download(d.download_bytes){}
    ~AuditCounters(){if(s){s->gpu_peak_bytes=std::max<uint64_t>(s->gpu_peak_bytes,d.shared?d.shared->peak:d.peak);s->gpu_allocations+=d.allocations-allocations;s->gpu_buffer_reuses+=d.reuses-reuses;s->gpu_upload_bytes+=d.upload_bytes-upload;s->gpu_download_bytes+=d.download_bytes-download;}}};
AuditCuda::AuditCuda(const NeuralOptions& options,MeshView source):impl_(std::make_unique<Impl>(options,source)){}
AuditCuda::~AuditCuda(){if(impl_){int previous=0;cudaGetDevice(&previous);impl_.reset();cudaSetDevice(previous);}}
Measurement AuditCuda::evaluate(MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& config,NeuralStats* stats) {
    CurrentDevice current_device(impl_->id);auto& device=impl_->device;
    AuditCounters counters(device,stats);
    if(auto e=validate(a);!e.empty())throw std::invalid_argument(e);if(auto e=validate(b);!e.empty())throw std::invalid_argument(e);
    // Validate evaluator settings through the identical-input fast path of the reference.
    (void)blitz::evaluate(a,a,bounds,config);Measurement result;result.supersample=config.supersample;
    if(config.cancelled&&config.cancelled()){result.complete=false;result.passed=false;return result;}
    if(same_mesh_data(a,b))return result;
    if(auto cached=impl_->memo.find(a,b,bounds,config)){if(stats)++stats->gpu_measurement_cache_hits;return *cached;}
    if(stats)++stats->gpu_evaluations;
    uint32_t view=0;uint8_t sampling=config.supersample;
    try {
    auto clear_images=[&]{impl_->clear_images();};
    if(impl_->source.positions.count&&!impl_->uploaded)impl_->uploaded=std::make_unique<UploadedMesh>(device,impl_->source);
    auto upload_mesh=[&](MeshView m,Impl::Topology& slot,std::unique_ptr<UploadedMesh>& owned)->const UploadedMesh*{
        if(impl_->uploaded&&source_attributes(m,impl_->source)){
            if(same_mesh_data(m,impl_->source))return impl_->uploaded.get();
            return slot.get(device,m,impl_->uploaded.get());
        }
        return slot.get(device,m,nullptr);
    };
    std::unique_ptr<UploadedMesh> owned_a,owned_b;
    const UploadedMesh *da=nullptr,*db=nullptr;
    try{da=upload_mesh(a,impl_->reference,owned_a);db=upload_mesh(b,impl_->candidate,owned_b);}
    catch(const ResourceError& error){if(error.kind!=NeuralResourceLimit::WorkspaceMemory&&error.kind!=NeuralResourceLimit::DeviceMemory)throw;
        clear_images();owned_a.reset();owned_b.reset();da=upload_mesh(a,impl_->reference,owned_a);db=upload_mesh(b,impl_->candidate,owned_b);}
    auto& images=impl_->source.positions.count&&same_mesh_data(a,impl_->source)?impl_->source_images:impl_->reference_images;
    images.configure(a,bounds,config);impl_->candidate_images.configure(b,bounds,config);
    result=impl_->run(*da,*db,bounds,config,images,stats,view,sampling);
    impl_->memo.insert(a,b,bounds,config,result);
    return result;
    }catch(const ResourceError& error){
        if(stats&&!stats->resource_failures++)stats->first_resource_failure={config.screen_size,error.requested,error.limit,view,sampling,error.kind};
        result.complete=false;result.passed=false;result.resource_limited=true;result.error=infinity;result.worst_view=view;result.supersample=sampling;return result;
    }
}
Measurement AuditCuda::evaluate(MeshView a,DeviceMeshView b,const Bounds& bounds,const EvalSettings& config,NeuralStats* stats,double incumbent,bool* pruned){
    if(pruned)*pruned=false;if(std::isfinite(incumbent)&&(!pruned||incumbent<0))throw std::invalid_argument("audit incumbent requires explicit pruning status");
    CurrentDevice current_device(impl_->id);auto& p=*impl_;auto& device=p.device;AuditCounters counters(device,stats);
    if(auto e=validate(a);!e.empty())throw std::invalid_argument(e);
    if(!b.identity||!b.positions||!b.indices||!b.vertices||!b.faces)throw std::invalid_argument("invalid borrowed device mesh");
    (void)blitz::evaluate(a,a,bounds,config);Measurement result;result.supersample=config.supersample;
    if(config.cancelled&&config.cancelled()){result.complete=false;result.passed=false;return result;}if(stats)++stats->gpu_evaluations;
    uint32_t view=0;uint8_t sampling=config.supersample;
    try{if(p.source.positions.count&&!p.uploaded)p.uploaded=std::make_unique<UploadedMesh>(device,p.source);
        std::unique_ptr<UploadedMesh> owned;const UploadedMesh* reference=nullptr;
        auto upload=[&]{if(p.uploaded&&source_attributes(a,p.source)){reference=same_mesh_data(a,p.source)?p.uploaded.get():p.reference.get(device,a,p.uploaded.get());}
            else reference=p.reference.get(device,a,nullptr);};
        try{upload();}catch(const ResourceError& error){if(error.kind!=NeuralResourceLimit::WorkspaceMemory&&error.kind!=NeuralResourceLimit::DeviceMemory)throw;p.clear_images();owned.reset();upload();}
        auto& images=p.source.positions.count&&same_mesh_data(a,p.source)?p.source_images:p.reference_images;
        images.configure(a,bounds,config);p.candidate_images.configure(b,bounds,config);
        return p.run(*reference,b,bounds,config,images,stats,view,sampling,incumbent,pruned);
    }catch(const ResourceError& error){if(stats&&!stats->resource_failures++)stats->first_resource_failure={config.screen_size,error.requested,error.limit,view,sampling,error.kind};
        result.complete=false;result.passed=false;result.resource_limited=true;result.error=infinity;result.worst_view=view;result.supersample=sampling;return result;}
}
}
namespace blitz {
Measurement evaluate_cuda(MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& config,const NeuralOptions& options,NeuralStats* stats) {
    neural::AuditCuda workspace(options,a);return workspace.evaluate(a,b,bounds,config,stats);
}
double overlap_cuda(MeshView m,const Bounds& bounds,double screen,ViewSet views,const NeuralOptions& options) {
    if(auto e=validate(m);!e.empty())throw std::invalid_argument(e);neural::gpu::Device device(options,true);neural::UploadedMesh uploaded(device,m);auto cameras_=cameras(bounds,screen,views);if(cameras_.empty())throw std::invalid_argument("overlap requires views");double overlap=0;
    neural::gpu::Buffer<neural::Summary> summary(device,1);
    for(auto c:cameras_){summary.zero();auto raster=neural::render(device,uploaded,bounds,c,screen,1,false,summary.p);if(raster.clipped)return std::numeric_limits<double>::infinity();auto s=summary.download()[0];overlap+=s.pixels?double(s.samples)/s.pixels:0;}
    return overlap/cameras_.size();
}
}
