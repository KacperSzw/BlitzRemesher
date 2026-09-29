#include "neural_cuda.cuh"
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
    Triangle* out,uint32_t* counts,uint32_t nf,uint32_t size,double diameter,bool perspective,bool force_two,int* clipped) {
    uint32_t f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;counts[f]=0;Triangle t{};
    auto ia=indices[f*3],ib=indices[f*3+1],ic=indices[f*3+2];t.a=verts[ia];t.b=verts[ib];t.c=verts[ic];t.material=materials?materials[f]:0;
    if(perspective&&(t.a.z<=0||t.b.z<=0||t.c.z<=0)){atomicExch(clipped,1);out[f]=t;return;}
    t.area=edge(t.a.x,t.a.y,t.b.x,t.b.y,t.c.x,t.c.y);t.back=t.area<0;
    if(fabs(t.area)<1e-16||(t.back&&!force_two&&!(t.material<ntwo&&two[t.material]))){out[f]=t;return;}
    auto p0=pos[ia],p1=pos[ib],p2=pos[ic];Vec3 d1{float((double(p1.x)-p0.x)/diameter),float((double(p1.y)-p0.y)/diameter),float((double(p1.z)-p0.z)/diameter)};
    Vec3 d2{float((double(p2.x)-p0.x)/diameter),float((double(p2.y)-p0.y)/diameter),float((double(p2.z)-p0.z)/diameter)};t.normal=norm(cp(d1,d2));
    if(t.back){auto b=t.b;t.b=t.c;t.c=b;t.area=-t.area;t.normal=mul(t.normal,-1);}
    double minx=fmin(t.a.x,fmin(t.b.x,t.c.x)),maxx=fmax(t.a.x,fmax(t.b.x,t.c.x));
    double miny=fmin(t.a.y,fmin(t.b.y,t.c.y)),maxy=fmax(t.a.y,fmax(t.b.y,t.c.y));
    if(!isfinite(minx)||!isfinite(maxx)||!isfinite(miny)||!isfinite(maxy)||minx<0||miny<0||maxx>size||maxy>size){atomicExch(clipped,1);out[f]=t;return;}
    t.x0=max(0,int(floor(minx)));t.x1=min(int(size)-1,int(floor(maxx)));t.y0=max(0,int(floor(miny)));t.y1=min(int(size)-1,int(floor(maxy)));
    t.r0=.5*(fabs(t.c.x-t.b.x)+fabs(t.c.y-t.b.y));t.r1=.5*(fabs(t.a.x-t.c.x)+fabs(t.a.y-t.c.y));t.r2=.5*(fabs(t.b.x-t.a.x)+fabs(t.b.y-t.a.y));t.valid=1;
    counts[f]=uint32_t((t.x1/16-t.x0/16+1)*(t.y1/16-t.y0/16+1));out[f]=t;
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
Image render(Device& device,MeshView m,const Bounds& b,const Camera& camera,double screen,uint8_t ss,bool two,Summary* summary=nullptr) {
    if(!ss||!std::isfinite(screen)||screen<=0||screen>16384)throw std::invalid_argument("invalid raster extent");
    uint32_t size=summary?(uint32_t(std::ceil(screen+8))+1)&~1u:uint32_t(std::ceil(screen+8))*ss;
    if(uint64_t(size)*size>max_raster_samples)throw ResourceError(NeuralResourceLimit::SampleCount,uint64_t(size)*size,max_raster_samples,"raster exceeds per-view sample cap");
    auto position=upload(device,m.positions),normals=upload(device,m.normals);auto colors=upload(device,m.colors);
    Buffer<uint32_t> indices(device,m.indices.size());indices.upload(m.indices);Buffer<uint16_t> materials(device,m.materials.size());materials.upload(m.materials);
    Buffer<uint8_t> double_sided(device,m.double_sided.size());double_sided.upload(m.double_sided);
    Buffer<Vertex> projected(device,m.positions.count);uint32_t nf=uint32_t(m.triangles());
    Buffer<Triangle> tris(device,nf);Buffer<uint32_t> counts(device,nf+1),offsets(device,nf+1);counts.zero();Buffer<int> clipped(device,1);clipped.zero();
    project<<<blocks(m.positions.count),256>>>(position.p,normals.p,colors.p,projected.p,uint32_t(m.positions.count),b,camera,size,ss);
    triangles<<<blocks(nf),256>>>(projected.p,position.p,indices.p,materials.p,double_sided.p,uint32_t(double_sided.n),tris.p,counts.p,nf,size,b.diameter(),camera.perspective,two,clipped.p);check(cudaGetLastError());
    // Bound all bin entries before the 32-bit scan. Counts are copied only once per view;
    // projected attributes, binning, sort and samples remain on the device.
    auto host_counts=counts.download();uint64_t entries=0;for(auto c:host_counts)entries+=c;
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
    bool was_clipped=clipped.download()[0]!=0;return {std::move(pixels),size,was_clipped};
}
__global__ void initialize(const Pixel* a,const Pixel* b,float* da,float* db,Summary* s,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;Summary local{};
    if(i<n){bool ca=a[i].covered,cb=b[i].covered;da[i]=ca?0.f:1e15f;db[i]=cb?0.f:1e15f;local.ca=ca;local.cb=cb;local.changed=ca!=cb;local.total=ca||cb;
        if(a[i].visible&&b[i].visible)local.normal=acos(fmin(1.,fmax(-1.,dp(a[i].normal,b[i].normal))))*180/3.14159265358979323846;}
    __shared__ cub::BlockReduce<Summary,256>::TempStorage storage;auto reduced=cub::BlockReduce<Summary,256>(storage).Reduce(local,AddSummary{});
    if(threadIdx.x==0){atomicAdd(&s->ca,reduced.ca);atomicAdd(&s->cb,reduced.cb);atomicAdd(&s->changed,reduced.changed);atomicAdd(&s->total,reduced.total);max_double(&s->normal,reduced.normal);}
}
// Exact separable squared Euclidean distance transform. One independent line per
// CUDA thread, with disjoint global stacks; no approximation or jump flooding.
__global__ void edt(const float* in,float* out,int* stack,double* boundaries,int size,bool columns) {
    int line=int(blockIdx.x*blockDim.x+threadIdx.x);if(line>=size)return;
    int* v=stack+size_t(line)*size;double* z=boundaries+size_t(line)*(size+1);int k=0;v[0]=0;z[0]=-CUDART_INF;z[1]=CUDART_INF;
    int stride=columns?size:1;size_t start=columns?size_t(line):size_t(line)*size;
    for(int q=1;q<size;++q){double s;for(;;){int p=v[k];s=((double(in[start+q*stride])+double(q)*q)-(double(in[start+p*stride])+double(p)*p))/(2*(q-p));if(s>z[k]||k==0)break;--k;}++k;v[k]=q;z[k]=s;z[k+1]=CUDART_INF;}
    k=0;for(int q=0;q<size;++q){while(z[k+1]<q)++k;double d=q-v[k];out[start+q*stride]=float(d*d+in[start+v[k]*stride]);}
}
__global__ void directed_coverage(const Pixel* from,const float* to,Summary* summary,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;double value=i<n&&from[i].covered?double(to[i]):0;
    __shared__ cub::BlockReduce<double,256>::TempStorage storage;double maximum=cub::BlockReduce<double,256>(storage).Reduce(value,cub::Max());
    if(threadIdx.x==0)max_double(&summary->distance,maximum);
}
__device__ double sample(const Pixel& a,const Pixel& b,double spatial,int profile,Weights weights) {
    double cost=spatial;
    if(profile&&weights.normal>0){double cosine=fmin(1.,fmax(-1.,dp(a.normal,b.normal)/fmax(1e-30,len(a.normal)*len(b.normal))));double angle=acos(cosine)*weights.normal;cost+=angle*angle;}
    if(profile==2){double x=double(a.color.x)-b.color.x,y=double(a.color.y)-b.color.y,z=double(a.color.z)-b.color.z;cost+=weights.color*weights.color*(x*x+y*y+z*z);if(a.material!=b.material)cost+=weights.material*weights.material;}return cost;
}
__global__ void appearance(const Pixel* from,const Pixel* to,Summary* summary,int size,int ss,double limit,int profile,Weights weights) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(size)*size||!from[i].visible)return;
    if(profile==0||(!weights.normal&&(profile!=2||(!weights.color&&!weights.material))))return;
    auto p=from[i];double best=to[i].visible?sample(p,to[i],0,profile,weights):CUDART_INF;if(best<=1e-18)return;
    int x=int(i%size),y=int(i/size),radius=int(fmin(double(size),ceil(limit*ss)));double limit2=limit*limit,invs2=1./(ss*ss);
    int r=int(fmin(double(radius),ceil(sqrt(fmin(best,limit2))*ss)));
    for(int yy=max(0,y-r);yy<=min(size-1,y+r);++yy)for(int xx=max(0,x-r);xx<=min(size-1,x+r);++xx){auto& q=to[size_t(yy)*size+xx];if(!q.visible)continue;double dx=x-xx,dy=y-yy,spatial=(dx*dx+dy*dy)*invs2;if(spatial>=best||spatial>limit2)continue;best=fmin(best,sample(p,q,spatial,profile,weights));}
    max_double(&summary->attribute,best>limit2?CUDART_INF:best);
}
Measurement measure(Device& device,MeshView a,MeshView b,const Bounds& bounds,const Camera& camera,const EvalSettings& config,uint8_t ss) {
    auto x=render(device,a,bounds,camera,config.screen_size,ss,config.force_two_sided),y=render(device,b,bounds,camera,config.screen_size,ss,config.force_two_sided);
    size_t n=x.pixels.n;int size=int(x.size);Buffer<float> da(device,n),db(device,n),temp(device,n);Buffer<int> stack(device,n);Buffer<double> boundaries(device,size_t(size)*(size+1));Buffer<Summary> summary(device,1);summary.zero();
    initialize<<<blocks(n),256>>>(x.pixels.p,y.pixels.p,da.p,db.p,summary.p,n);check(cudaGetLastError());auto s=summary.download()[0];
    if(s.changed&&bool(s.ca)==bool(s.cb)) {
        for(auto pair:{std::pair{da.p,y.pixels.p},std::pair{db.p,x.pixels.p}}) {
            edt<<<blocks(size),256>>>(pair.first,temp.p,stack.p,boundaries.p,size,false);edt<<<blocks(size),256>>>(temp.p,pair.first,stack.p,boundaries.p,size,true);
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
    if(auto error=validate(m);!error.empty())throw std::invalid_argument(error);Device device(options);auto image=render(device,m,b,c,screen,ss,two);return {image.size,image.size,image.pixels.download(),image.clipped};
}
}
namespace blitz {
Measurement evaluate_cuda(MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& config,const NeuralOptions& options,NeuralStats* stats) {
    if(auto e=validate(a);!e.empty())throw std::invalid_argument(e);if(auto e=validate(b);!e.empty())throw std::invalid_argument(e);
    // Validate evaluator settings through the identical-input fast path of the reference.
    (void)evaluate(a,a,bounds,config);Measurement result;result.supersample=config.supersample;if(same_mesh_data(a,b))return result;
    neural::gpu::Device device(options);auto views=cameras(bounds,config.screen_size,config.views);
    for(uint32_t v=0;v<views.size();++v) {
        if(config.cancelled&&config.cancelled()){result.complete=false;result.passed=false;return result;}
        Measurement current;
        for(unsigned ss=config.supersample;;ss=std::min<unsigned>(config.max_supersample,ss*2)) {
            try{current=neural::measure(device,a,b,bounds,views[v],config,uint8_t(ss));}
            catch(const neural::gpu::ResourceError& error){
                if(stats){stats->gpu_peak_bytes=std::max<uint64_t>(stats->gpu_peak_bytes,device.peak);
                    if(!stats->resource_failures++)stats->first_resource_failure={config.screen_size,error.requested,error.limit,v,uint8_t(ss),error.kind};}
                result.complete=false;result.passed=false;result.resource_limited=true;result.error=std::numeric_limits<double>::infinity();result.worst_view=v;result.supersample=uint8_t(ss);return result;
            }
            if(stats)stats->gpu_peak_bytes=std::max<uint64_t>(stats->gpu_peak_bytes,device.peak);
            if(current.passed||ss>=config.max_supersample||current.coverage-2*std::sqrt(2.)/ss>config.limit||!std::isfinite(current.error)||current.coverage_upper<=config.limit)break;
        }
        ++result.views_evaluated;if(current.error>result.error){result.worst_view=v;result.error=current.error;result.supersample=current.supersample;}
        result.coverage=std::max(result.coverage,current.coverage);result.coverage_upper=std::max(result.coverage_upper,current.coverage_upper);result.normal_degrees=std::max(result.normal_degrees,current.normal_degrees);
        if(current.changed_area>result.changed_area){result.changed_area=current.changed_area;result.changed_area_worst_view=v;}
        if(!current.passed){result.passed=false;result.complete=false;return result;}
    }
    return result;
}
double overlap_cuda(MeshView m,const Bounds& bounds,double screen,ViewSet views,const NeuralOptions& options) {
    if(auto e=validate(m);!e.empty())throw std::invalid_argument(e);neural::gpu::Device device(options);auto cameras_=cameras(bounds,screen,views);if(cameras_.empty())throw std::invalid_argument("overlap requires views");double overlap=0;
    neural::gpu::Buffer<neural::Summary> summary(device,1);
    for(auto c:cameras_){summary.zero();auto raster=neural::render(device,m,bounds,c,screen,1,false,summary.p);if(raster.clipped)return std::numeric_limits<double>::infinity();auto s=summary.download()[0];overlap+=s.pixels?double(s.samples)/s.pixels:0;}
    return overlap/cameras_.size();
}
}
