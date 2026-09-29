#include "neural_cuda.cuh"
#include "neural_audit_cache.hpp"
#include "metric_angle.hpp"
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
struct Triangle {Vertex a,b,c;Vec3 normal;double area,r0,r1,r2;int x0,x1,y0,y1;uint16_t material;uint8_t back,valid;};
struct Bins {unsigned long long entries;int clipped;};
struct Summary {unsigned long long ca,cb,changed,total,samples,pixels;double distance,attribute,normal;};
struct AddSummary {__device__ Summary operator()(Summary a,Summary b) const {return {a.ca+b.ca,a.cb+b.cb,a.changed+b.changed,a.total+b.total,a.samples+b.samples,a.pixels+b.pixels,fmax(a.distance,b.distance),fmax(a.attribute,b.attribute),fmax(a.normal,b.normal)};}};
__device__ void max_double(double* p,double x){atomicMax(reinterpret_cast<unsigned long long*>(p),__double_as_longlong(x));}
__global__ void project(const Vec3* position,const Vec3* normals,const ColorRGBA8* colors,Vertex* out,uint32_t n,Bounds bounds,Camera camera,uint32_t size,uint8_t ss) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;auto a=position[i];double x=double(a.x)-bounds.center.x,y=double(a.y)-bounds.center.y,z0=double(a.z)-bounds.center.z;
    double z=camera.distance-(x*camera.forward.x+y*camera.forward.y+z0*camera.forward.z),scale=camera.perspective?camera.focal/z:camera.scale;
    Vertex p;p.x=(x*camera.right.x+y*camera.right.y+z0*camera.right.z)*scale*ss+size*.5;p.y=(x*camera.up.x+y*camera.up.y+z0*camera.up.z)*scale*ss+size*.5;p.z=z;
    p.normal=norm(normals?normals[i]:Vec3{});p.color=colors?Vec4{colors[i].r/255.f,colors[i].g/255.f,colors[i].b/255.f,colors[i].a/255.f}:Vec4{1,1,1,1};out[i]=p;
}
__global__ void triangles(const Vertex* verts,const Vec3* pos,const uint32_t* indices,const uint16_t* materials,const uint8_t* two,uint32_t ntwo,
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
__global__ void raster(const Triangle* tris,const uint64_t* keys,const uint32_t* begin,const uint32_t* end,Pixel* pixels,uint32_t size,
    uint32_t tiles,bool perspective,double diameter,Summary* summary) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(size)*size)return;int x=int(i%size),y=int(i/size);auto tile=uint32_t(y/16)*tiles+x/16;
    Pixel p{};p.depth=CUDART_INF_F;unsigned long long overlap=0;
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
        if(depth>=p.depth)continue;p.depth=float(depth);p.visible=1;p.material=t.material;
        p.normal=norm(add(add(mul(t.a.normal,u),mul(t.b.normal,v)),mul(t.c.normal,w)));
        if(len(p.normal)<.5)p.normal=t.normal;else if(t.back)p.normal=mul(p.normal,-1);
        p.color={float(u*t.a.color.x+v*t.b.color.x+w*t.c.color.x),float(u*t.a.color.y+v*t.b.color.y+w*t.c.color.y),float(u*t.a.color.z+v*t.b.color.z+w*t.c.color.z),1};
    }
    pixels[i]=p;if(summary&&overlap){atomicAdd(&summary->samples,overlap);atomicAdd(&summary->pixels,1ull);}
}
template<class T> Buffer<T> upload(Device& device,Stream<T> stream) {
    Buffer<T> b(device,stream.count);if(stream.stride==sizeof(T))b.upload({reinterpret_cast<const T*>(stream.data),stream.count});
    else {std::vector<T> packed(stream.count);for(size_t i=0;i<stream.count;++i)packed[i]=stream[i];b.upload(packed);}return b;
}
struct Image {Buffer<Pixel> pixels;uint32_t size;bool clipped;};
struct UploadedMesh {
    Buffer<Vec3> positions,normals;Buffer<ColorRGBA8> colors;
    Buffer<uint32_t> indices;Buffer<uint16_t> materials;Buffer<uint8_t> double_sided;
    const Vec3 *position,*normal;const ColorRGBA8* color;const uint8_t* sided;
    uint32_t vertices,faces,sided_count;
    UploadedMesh(Device& d,MeshView m,const UploadedMesh* shared=nullptr):
        positions(shared?Buffer<Vec3>{}:upload(d,m.positions)),normals(shared?Buffer<Vec3>{}:upload(d,m.normals)),
        colors(shared?Buffer<ColorRGBA8>{}:upload(d,m.colors)),indices(d,m.indices.size()),materials(d,m.materials.size()),double_sided(d,shared?0:m.double_sided.size()),
        position(shared?shared->position:positions.p),normal(shared?shared->normal:normals.p),color(shared?shared->color:colors.p),sided(shared?shared->sided:double_sided.p),
        vertices(uint32_t(m.positions.count)),faces(uint32_t(m.triangles())),sided_count(uint32_t(m.double_sided.size())) {
        indices.upload(m.indices);materials.upload(m.materials);if(!shared)double_sided.upload(m.double_sided);
    }
};
Image render(Device& device,const UploadedMesh& m,const Bounds& b,const Camera& camera,double screen,uint8_t ss,bool two,Summary* summary=nullptr) {
    if(!ss||!std::isfinite(screen)||screen<=0||screen>16384)throw std::invalid_argument("invalid raster extent");
    uint32_t size=summary?(uint32_t(std::ceil(screen+8))+1)&~1u:uint32_t(std::ceil(screen+8))*ss;
    if(uint64_t(size)*size>max_raster_samples)throw ResourceError(NeuralResourceLimit::SampleCount,uint64_t(size)*size,max_raster_samples,"raster exceeds per-view sample cap");
    Buffer<Vertex> projected(device,m.vertices);uint32_t nf=m.faces;
    Buffer<Triangle> tris(device,nf);Buffer<uint32_t> counts(device,nf+1),offsets(device,nf+1);counts.zero();Buffer<Bins> bins(device,1);bins.zero();
    project<<<blocks(m.vertices),256>>>(m.position,m.normal,m.color,projected.p,m.vertices,b,camera,size,ss);
    triangles<<<blocks(nf),256>>>(projected.p,m.position,m.indices.p,m.materials.p,m.sided,m.sided_count,tris.p,counts.p,bins.p,nf,size,b.diameter(),camera.perspective,two);check(cudaGetLastError());
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
    Buffer<Pixel> pixels(device,size_t(size)*size);
    raster<<<blocks(pixels.n),256>>>(tris.p,sorted.p,begin.p,end.p,pixels.p,size,tile_width,camera.perspective,b.diameter(),summary);check(cudaGetLastError());
    return {std::move(pixels),size,bin_result.clipped!=0};
}
__global__ void initialize(const Pixel* a,const Pixel* b,float* da,float* db,Summary* s,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;Summary local{};
    if(i<n){bool ca=a[i].covered,cb=b[i].covered;da[i]=ca?0.f:1e15f;db[i]=cb?0.f:1e15f;local.ca=ca;local.cb=cb;local.changed=ca!=cb;local.total=ca||cb;
        if(a[i].visible&&b[i].visible)local.normal=detail::metric_acos(fmin(1.,fmax(-1.,dp(a[i].normal,b[i].normal))))*180/3.14159265358979323846;}
    __shared__ cub::BlockReduce<Summary,256>::TempStorage storage;auto reduced=cub::BlockReduce<Summary,256>(storage).Reduce(local,AddSummary{});
    if(threadIdx.x==0){atomicAdd(&s->ca,reduced.ca);atomicAdd(&s->cb,reduced.cb);atomicAdd(&s->changed,reduced.changed);atomicAdd(&s->total,reduced.total);max_double(&s->normal,reduced.normal);}
}
// Transposing lets adjacent lanes visit adjacent pixels in both separable
// passes. The padded shared tile also avoids bank conflicts on the write pass.
__global__ void transpose(const float* in,float* out,int size) {
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
__global__ void edt_binary(const float* in,float* out,int size) {
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
__global__ void edt(const float* in,float* out,int* stack,double* boundaries,int size) {
    int line=int(blockIdx.x*blockDim.x+threadIdx.x);if(line>=size)return;
    int* v=stack+line;double* z=boundaries+line;int k=0;v[0]=0;z[0]=-CUDART_INF;z[size]=CUDART_INF;
    for(int q=1;q<size;++q){double s;for(;;){int p=v[size_t(k)*size];s=((double(in[size_t(q)*size+line])+double(q)*q)-(double(in[size_t(p)*size+line])+double(p)*p))/(2*(q-p));if(s>z[size_t(k)*size]||k==0)break;--k;}++k;v[size_t(k)*size]=q;z[size_t(k)*size]=s;z[size_t(k+1)*size]=CUDART_INF;}
    k=0;for(int q=0;q<size;++q){while(z[size_t(k+1)*size]<q)++k;int p=v[size_t(k)*size];double d=q-p;out[size_t(q)*size+line]=float(d*d+in[size_t(p)*size+line]);}
}
__global__ void directed_coverage(const Pixel* from,const float* to,Summary* summary,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double value=i<n&&from[i].covered?double(to[i]):0;
    __shared__ cub::BlockReduce<double,256>::TempStorage storage;double maximum=cub::BlockReduce<double,256>(storage).Reduce(value,cub::Max());
    if(threadIdx.x==0)max_double(&summary->distance,maximum);
}
__device__ double sample(const Pixel& a,const Pixel& b,double spatial,int profile,Weights weights) {
    double cost=spatial;
    if(profile&&weights.normal>0){double cosine=fmin(1.,fmax(-1.,dp(a.normal,b.normal)/fmax(1e-30,len(a.normal)*len(b.normal))));double angle=detail::metric_acos(cosine)*weights.normal;cost+=angle*angle;}
    if(profile==2){double x=double(a.color.x)-b.color.x,y=double(a.color.y)-b.color.y,z=double(a.color.z)-b.color.z;cost+=weights.color*weights.color*(x*x+y*y+z*z);if(a.material!=b.material)cost+=weights.material*weights.material;}return cost;
}
__device__ double appearance_pixel(const Pixel* from,const Pixel* to,size_t i,int size,int ss,double limit,int profile,Weights weights) {
    if(i>=size_t(size)*size||!from[i].visible)return 0;
    if(profile==0||(!weights.normal&&(profile!=2||(!weights.color&&!weights.material))))return 0;
    auto p=from[i];double best=to[i].visible?sample(p,to[i],0,profile,weights):CUDART_INF;if(best<=1e-18)return 0;
    int x=int(i%size),y=int(i/size),radius=int(fmin(double(size),ceil(limit*ss)));double limit2=limit*limit,invs2=1./(ss*ss);
    // Any other sample has spatial cost at least invs2 and nonnegative attribute
    // cost. Keep the computed center cost (including its rounding), but avoid a
    // search that cannot improve it. The center itself was already evaluated.
    if(best<=invs2)return best>limit2?CUDART_INF:best;
    int r=int(fmin(double(radius),ceil(sqrt(fmin(best,limit2))*ss)));
    for(int yy=max(0,y-r);yy<=min(size-1,y+r);++yy)for(int xx=max(0,x-r);xx<=min(size-1,x+r);++xx){if(xx==x&&yy==y)continue;auto& q=to[size_t(yy)*size+xx];if(!q.visible)continue;double dx=x-xx,dy=y-yy,spatial=(dx*dx+dy*dy)*invs2;if(spatial>=best||spatial>limit2)continue;best=fmin(best,sample(p,q,spatial,profile,weights));}
    return best>limit2?CUDART_INF:best;
}
__global__ void appearance(const Pixel* from,const Pixel* to,Summary* summary,int size,int ss,double limit,int profile,Weights weights) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    double value=appearance_pixel(from,to,i,size,ss,limit,profile,weights);
    // Every lane participates, including invisible/out-of-image pixels. Preserve
    // exact nonnegative maxima while issuing only one global atomic per block.
    __shared__ cub::BlockReduce<double,256>::TempStorage storage;
    double maximum=cub::BlockReduce<double,256>(storage).Reduce(value,cub::Max());
    if(threadIdx.x==0&&maximum>0)max_double(&summary->attribute,maximum);
}
Measurement measure(Device& device,const UploadedMesh& a,const UploadedMesh& b,const Bounds& bounds,const Camera& camera,const EvalSettings& config,uint8_t ss) {
    auto x=render(device,a,bounds,camera,config.screen_size,ss,config.force_two_sided),y=render(device,b,bounds,camera,config.screen_size,ss,config.force_two_sided);
    size_t n=x.pixels.n;int size=int(x.size);Buffer<float> da(device,n),db(device,n),temp(device,n);Buffer<int> stack(device,n);Buffer<double> boundaries(device,size_t(size)*(size+1));Buffer<Summary> summary(device,1);summary.zero();
    initialize<<<blocks(n),256>>>(x.pixels.p,y.pixels.p,da.p,db.p,summary.p,n);check(cudaGetLastError());auto s=summary.download()[0];
    if(s.changed&&bool(s.ca)==bool(s.cb)) {
        for(auto pair:{std::pair{da.p,y.pixels.p},std::pair{db.p,x.pixels.p}}) {
            dim3 tiles((size+31)/32,(size+31)/32),threads(32,8);
            transpose<<<tiles,threads>>>(pair.first,temp.p,size);
            edt_binary<<<(size+63)/64,64>>>(temp.p,pair.first,size);
            transpose<<<tiles,threads>>>(pair.first,temp.p,size);
            edt<<<(size+63)/64,64>>>(temp.p,pair.first,stack.p,boundaries.p,size);
            directed_coverage<<<blocks(n),256>>>(pair.second,pair.first,summary.p,n);check(cudaGetLastError());
        }
        s=summary.download()[0];
    }
    Measurement m;m.supersample=ss;m.coverage=bool(s.ca)!=bool(s.cb)?infinity:std::sqrt(s.distance)/ss;
    m.coverage_upper=m.coverage+2*std::sqrt(2.)/ss+1e-6;if(x.clipped||y.clipped)m.coverage_upper=infinity;
    m.changed_area=s.total?double(s.changed)/s.total:0;m.error=m.coverage_upper;m.normal_degrees=config.profile==Profile::Coverage?0:s.normal;
    if(m.coverage_upper<=config.limit&&config.profile!=Profile::Coverage) {
        appearance<<<blocks(n),256>>>(x.pixels.p,y.pixels.p,summary.p,size,ss,config.limit,int(config.profile),config.weights);
        appearance<<<blocks(n),256>>>(y.pixels.p,x.pixels.p,summary.p,size,ss,config.limit,int(config.profile),config.weights);check(cudaGetLastError());s=summary.download()[0];m.error=std::max(m.error,std::sqrt(s.attribute));
    }
    m.passed=m.error<=config.limit&&m.changed_area<=config.max_changed_area;return m;
}
}
Raster raster_cuda(MeshView m,const Bounds& b,const Camera& c,double screen,uint8_t ss,bool two,const NeuralOptions& options) {
    if(auto error=validate(m);!error.empty())throw std::invalid_argument(error);Device device(options,true);UploadedMesh uploaded(device,m);auto image=render(device,uploaded,b,c,screen,ss,two);return {image.size,image.size,image.pixels.download(),image.clipped};
}
struct AuditCuda::Impl {
    struct Topology {
        std::vector<uint32_t> indices;std::vector<uint16_t> materials;std::unique_ptr<UploadedMesh> uploaded;
        bool matches(MeshView m) const {return uploaded&&std::equal(indices.begin(),indices.end(),m.indices.begin(),m.indices.end())&&std::equal(materials.begin(),materials.end(),m.materials.begin(),m.materials.end());}
        const UploadedMesh* get(Device& device,MeshView m,const UploadedMesh* source){
            if(!matches(m)){uploaded.reset();indices.assign(m.indices.begin(),m.indices.end());materials.assign(m.materials.begin(),m.materials.end());uploaded=std::make_unique<UploadedMesh>(device,m,source);}
            return uploaded.get();
        }
    };
    Device device;int id;MeshView source;AuditMemo memo;std::unique_ptr<UploadedMesh> uploaded;
    Topology reference,candidate;
    Impl(const NeuralOptions& options,MeshView mesh):device(options,true),id(options.device),source(mesh),memo(mesh){check(cudaSetDevice(device.previous));}
    ~Impl(){cudaSetDevice(id);}
};
struct CurrentDevice {int previous;explicit CurrentDevice(int id){check(cudaGetDevice(&previous));check(cudaSetDevice(id));}~CurrentDevice(){cudaSetDevice(previous);}};
AuditCuda::AuditCuda(const NeuralOptions& options,MeshView source):impl_(std::make_unique<Impl>(options,source)){}
AuditCuda::~AuditCuda(){if(impl_){int previous=0;cudaGetDevice(&previous);impl_.reset();cudaSetDevice(previous);}}
Measurement AuditCuda::evaluate(MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& config,NeuralStats* stats) {
    CurrentDevice current_device(impl_->id);auto& device=impl_->device;
    struct Counters {Device& d;NeuralStats* s;uint64_t allocations,reuses,upload,download;
        ~Counters(){if(s){s->gpu_peak_bytes=std::max<uint64_t>(s->gpu_peak_bytes,d.peak);s->gpu_allocations+=d.allocations-allocations;s->gpu_buffer_reuses+=d.reuses-reuses;s->gpu_upload_bytes+=d.upload_bytes-upload;s->gpu_download_bytes+=d.download_bytes-download;}}
    } counters{device,stats,device.allocations,device.reuses,device.upload_bytes,device.download_bytes};
    if(auto e=validate(a);!e.empty())throw std::invalid_argument(e);if(auto e=validate(b);!e.empty())throw std::invalid_argument(e);
    // Validate evaluator settings through the identical-input fast path of the reference.
    (void)blitz::evaluate(a,a,bounds,config);Measurement result;result.supersample=config.supersample;
    if(config.cancelled&&config.cancelled()){result.complete=false;result.passed=false;return result;}
    if(same_mesh_data(a,b))return result;
    if(auto cached=impl_->memo.find(a,b,bounds,config)){if(stats)++stats->gpu_measurement_cache_hits;return *cached;}
    if(stats)++stats->gpu_evaluations;
    uint32_t view=0;uint8_t sampling=config.supersample;
    try {
    if(impl_->source.positions.count&&!impl_->uploaded)impl_->uploaded=std::make_unique<UploadedMesh>(device,impl_->source);
    auto upload_mesh=[&](MeshView m,Impl::Topology& slot,std::unique_ptr<UploadedMesh>& owned)->const UploadedMesh*{
        if(impl_->uploaded&&source_attributes(m,impl_->source)){
            if(same_mesh_data(m,impl_->source))return impl_->uploaded.get();
            return slot.get(device,m,impl_->uploaded.get());
        }
        owned=std::make_unique<UploadedMesh>(device,m);return owned.get();
    };
    std::unique_ptr<UploadedMesh> owned_a,owned_b;
    const auto* da=upload_mesh(a,impl_->reference,owned_a);
    const auto* db=upload_mesh(b,impl_->candidate,owned_b);
    auto views=cameras(bounds,config.screen_size,config.views);
    for(uint32_t v=0;v<views.size();++v) {
        view=v;
        if(config.cancelled&&config.cancelled()){result.complete=false;result.passed=false;return result;}
        Measurement current;
        for(unsigned ss=config.supersample;;ss=std::min<unsigned>(config.max_supersample,ss*2)) {
            sampling=uint8_t(ss);current=measure(device,*da,*db,bounds,views[v],config,sampling);
            if(stats)stats->gpu_peak_bytes=std::max<uint64_t>(stats->gpu_peak_bytes,device.peak);
            if(current.passed||ss>=config.max_supersample||current.coverage-2*std::sqrt(2.)/ss>config.limit||!std::isfinite(current.error)||current.coverage_upper<=config.limit)break;
        }
        ++result.views_evaluated;if(current.error>result.error){result.worst_view=v;result.error=current.error;result.supersample=current.supersample;}
        result.coverage=std::max(result.coverage,current.coverage);result.coverage_upper=std::max(result.coverage_upper,current.coverage_upper);result.normal_degrees=std::max(result.normal_degrees,current.normal_degrees);
        if(current.changed_area>result.changed_area){result.changed_area=current.changed_area;result.changed_area_worst_view=v;}
        if(!current.passed){result.passed=false;result.complete=false;return result;}
    }
    impl_->memo.insert(a,b,bounds,config,result);
    return result;
    }catch(const ResourceError& error){
        if(stats&&!stats->resource_failures++)stats->first_resource_failure={config.screen_size,error.requested,error.limit,view,sampling,error.kind};
        result.complete=false;result.passed=false;result.resource_limited=true;result.error=infinity;result.worst_view=view;result.supersample=sampling;return result;
    }
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
