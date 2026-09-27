#include "blitz/evaluate.hpp"
#include <stdexcept>
#include <numeric>
#if defined(BLITZ_AVX2) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#endif
namespace blitz {
namespace {
constexpr double pi=3.14159265358979323846, inf=std::numeric_limits<double>::infinity();
Vec3 rotate(Vec3 v,uint32_t seed) {
    double a=(seed%65521)*.000731,b=((seed>>8)%32749)*.000913;
    double x=v.x*std::cos(a)-v.y*std::sin(a),y=v.x*std::sin(a)+v.y*std::cos(a);
    return {float(x),float(y*std::cos(b)-v.z*std::sin(b)),float(y*std::sin(b)+v.z*std::cos(b))};
}
double edge(double ax,double ay,double bx,double by,double x,double y) {return (bx-ax)*(y-ay)-(by-ay)*(x-ax);}
#if defined(BLITZ_AVX2) && defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
__attribute__((target("avx2"))) void edt_span(float* out,int begin,int end,int center,double value) {
    int q=begin;
    for(;q+4<=end;q+=4) {
        auto x=_mm256_setr_pd(double(q-center),double(q+1-center),double(q+2-center),double(q+3-center));
        auto d=_mm256_add_pd(_mm256_mul_pd(x,x),_mm256_set1_pd(value));
        _mm_storeu_ps(out+q,_mm256_cvtpd_ps(d));
    }
    for(;q<end;++q){double d=q-center;out[q]=float(d*d+value);}
}
#endif
void edt_line(const float* f,float* out,int n,std::vector<int>& v,std::vector<double>& z,bool scalar) {
    int k=0;v[0]=0;z[0]=-inf;z[1]=inf;
    for(int q=1;q<n;++q) {
        double s;
        for(;;) {int p=v[k];s=((double(f[q])+double(q)*q)-(double(f[p])+double(p)*p))/(2*(q-p));if(s>z[k]||k==0)break;--k;}
        ++k;v[k]=q;z[k]=s;z[k+1]=inf;
    }
    k=0;
    for(int q=0;q<n;) {
        while(z[k+1]<q)++k;
        int end=z[k+1]>=n?n:std::max(q+1,int(std::floor(z[k+1]))+1);
#if defined(BLITZ_AVX2) && defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
        if(!scalar&&end-q>=8&&__builtin_cpu_supports("avx2"))edt_span(out,q,end,v[k],f[v[k]]);
        else
#else
        (void)scalar;
#endif
        for(int x=q;x<end;++x){double d=x-v[k];out[x]=float(d*d+f[v[k]]);}
        q=end;
    }
}
std::vector<float> distance_field(const Raster& r,bool scalar) {
    size_t n=r.pixels.size();std::vector<float> a(n),b(n);
    for(size_t i=0;i<n;++i)a[i]=r.pixels[i].covered?0.f:1e15f;
    size_t len=std::max(r.width,r.height);std::vector<int> v(len);std::vector<double> z(len+1);
    std::vector<float> f(len),d(len);
    for(uint32_t y=0;y<r.height;++y)edt_line(a.data()+size_t(y)*r.width,b.data()+size_t(y)*r.width,int(r.width),v,z,scalar);
    for(uint32_t x=0;x<r.width;++x) {
        for(uint32_t y=0;y<r.height;++y)f[y]=b[size_t(y)*r.width+x];
        edt_line(f.data(),d.data(),int(r.height),v,z,scalar);
        for(uint32_t y=0;y<r.height;++y)a[size_t(y)*r.width+x]=d[y];
    }
    return a;
}
template<class T> bool equal_stream(Stream<T> a,Stream<T> b) {
    if(a.count!=b.count)return false;if(!a.count)return true;
    if(a.data==b.data&&a.stride==b.stride)return true;
    if(a.stride==sizeof(T)&&b.stride==sizeof(T))return std::memcmp(a.data,b.data,a.count*sizeof(T))==0;
    for(size_t i=0;i<a.count;++i){auto x=a[i],y=b[i];if(std::memcmp(&x,&y,sizeof(T)))return false;}return true;
}
bool identical(MeshView a,MeshView b) {
    return equal_stream(a.positions,b.positions)&&equal_stream(a.normals,b.normals)&&equal_stream(a.colors,b.colors)
      &&std::equal(a.indices.begin(),a.indices.end(),b.indices.begin(),b.indices.end())
      &&std::equal(a.materials.begin(),a.materials.end(),b.materials.begin(),b.materials.end())
      &&std::equal(a.double_sided.begin(),a.double_sided.end(),b.double_sided.begin(),b.double_sided.end());
}
double sample_cost(const Pixel& a,const Pixel& b,double spatial,const EvalSettings& s) {
    double cost=spatial;
    if(s.profile!=Profile::Coverage && s.weights.normal>0) {
        double cosine=std::clamp(dot(a.normal,b.normal)/std::max(1e-30,length(a.normal)*length(b.normal)),-1.0,1.0);
        double angle=std::acos(cosine)*s.weights.normal;cost+=angle*angle;
    }
    if(s.profile==Profile::Attributes) {
        double x=double(a.color.x)-b.color.x,y=double(a.color.y)-b.color.y,z=double(a.color.z)-b.color.z;
        cost+=s.weights.color*s.weights.color*(x*x+y*y+z*z);
        if(a.material!=b.material)cost+=s.weights.material*s.weights.material;
    }
    return cost;
}
double spatial_scalar(int x,int y,int xx,int yy,double invs2) {
    double dx=x-xx,dy=y-yy;return (dx*dx+dy*dy)*invs2;
}
}
const char* evaluator_backend(bool force_scalar) {
#if defined(BLITZ_AVX2) && defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    if(!force_scalar&&__builtin_cpu_supports("avx2"))return "avx2";
#else
    (void)force_scalar;
#endif
    return "scalar";
}
std::vector<Camera> cameras(const Bounds& b,double screen,ViewSet set) {
    std::vector<Camera> out;out.reserve(size_t(set.orthographic)+set.perspective);
    for(int projection=0;projection<2;++projection) {
        unsigned count=projection?set.perspective:set.orthographic;
        for(unsigned i=0;i<count;++i) {
            double z=1-2*(i+.5)/count,r=std::sqrt(std::max(0.0,1-z*z)),angle=i*pi*(3-std::sqrt(5.0));
            Vec3 forward=normalized(rotate({float(r*std::cos(angle)),float(z),float(r*std::sin(angle))},set.rotation_seed+projection*104729u));
            Vec3 hint=std::abs(forward.y)>.99?Vec3{1,0,0}:Vec3{0,1,0};
            Vec3 right=normalized(cross(hint,forward)),up=normalized(cross(forward,right));
            // Fixed 60 degree vertical FOV; a virtual viewport >=2*S frames the entire sphere.
            double height=std::max(1024.0,2*screen),focal=height/(2*std::tan(pi/6));
            double distance=b.radius*std::sqrt(1+std::pow(2*focal/screen,2));
            out.push_back({right,up,forward,distance,focal,screen/b.diameter(),bool(projection)});
        }
    }
    return out;
}
Raster rasterize(MeshView m,const Bounds& b,const Camera& c,double screen,uint8_t ss,bool force_two) {
    if(!ss||!std::isfinite(screen)||screen<=0||screen>16384)throw std::invalid_argument("invalid raster extent");
    uint32_t size=uint32_t(std::ceil(screen+8))*ss;
    if(uint64_t(size)*size>64000000)throw std::length_error("raster exceeds per-view memory budget");
    Raster out{size,size,std::vector<Pixel>(size_t(size)*size)};
    struct P {double x,y,z;Vec3 n;Vec4 col;};
    std::vector<P> projected(m.positions.count);
    for(size_t i=0;i<m.positions.count;++i) {
        auto a=m.positions[i];double x=double(a.x)-b.center.x,y=double(a.y)-b.center.y,z0=double(a.z)-b.center.z;
        auto dp=[&](Vec3 axis){return x*axis.x+y*axis.y+z0*axis.z;};
        double z=c.distance-dp(c.forward),scale=c.perspective?c.focal/z:c.scale;
        projected[i]={dp(c.right)*scale*ss+size*.5,dp(c.up)*scale*ss+size*.5,z,
          m.normals?normalized(m.normals[i]):Vec3{},m.colors?m.colors[i]:Vec4{1,1,1,1}};
    }
    for(size_t f=0;f<m.triangles();++f) {
        auto ia=m.indices[3*f],ib=m.indices[3*f+1],ic=m.indices[3*f+2];
        P a=projected[ia],d=projected[ib],e=projected[ic];
        if(c.perspective&&(a.z<=0||d.z<=0||e.z<=0)){out.clipped=true;continue;}
        double area=edge(a.x,a.y,d.x,d.y,e.x,e.y);
        if(std::abs(area)<1e-16)continue;
        bool back=area<0,two=force_two||m.two_sided(f);
        if(back&&!two)continue;
        auto delta=[&](uint32_t i){auto p=m.positions[i],p0=m.positions[ia];double scale=b.diameter();
            return Vec3{float((double(p.x)-p0.x)/scale),float((double(p.y)-p0.y)/scale),float((double(p.z)-p0.z)/scale)};};
        Vec3 geometric=normalized(cross(delta(ib),delta(ic)));
        if(back){std::swap(d,e);area=-area;geometric=geometric*-1;}
        double minx=std::min({a.x,d.x,e.x}),maxx=std::max({a.x,d.x,e.x});
        double miny=std::min({a.y,d.y,e.y}),maxy=std::max({a.y,d.y,e.y});
        if(!std::isfinite(minx)||!std::isfinite(maxx)||!std::isfinite(miny)||!std::isfinite(maxy)
           ||minx<0||miny<0||maxx>size||maxy>size){out.clipped=true;return out;}
        int x0=std::max(0,int(std::floor(minx))),x1=std::min(int(size)-1,int(std::floor(maxx)));
        int y0=std::max(0,int(std::floor(miny))),y1=std::min(int(size)-1,int(std::floor(maxy)));
        double r0=.5*(std::abs(e.x-d.x)+std::abs(e.y-d.y));
        double r1=.5*(std::abs(a.x-e.x)+std::abs(a.y-e.y));
        double r2=.5*(std::abs(d.x-a.x)+std::abs(d.y-a.y));
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x) {
            double u=edge(d.x,d.y,e.x,e.y,x+.5,y+.5),v=edge(e.x,e.y,a.x,a.y,x+.5,y+.5),w=edge(a.x,a.y,d.x,d.y,x+.5,y+.5);
            if(u+r0<0||v+r1<0||w+r2<0)continue;
            auto& pixel=out.pixels[size_t(y)*size+x];pixel.covered=1;
            if(u<0||v<0||w<0)continue;
            u/=area;v/=area;w/=area;
            double depth;
            if(c.perspective){double sum=u/a.z+v/d.z+w/e.z;depth=1/sum;u=u/a.z/sum;v=v/d.z/sum;w=w/e.z/sum;}
            else depth=u*a.z+v*d.z+w*e.z;
            depth/=b.diameter();
            if(depth>=pixel.depth)continue;
            pixel.depth=float(depth);pixel.visible=1;pixel.material=m.material(f);
            pixel.normal=normalized(a.n*u+d.n*v+e.n*w);
            if(length(pixel.normal)<.5)pixel.normal=geometric;
            else if(back)pixel.normal=pixel.normal*-1;
            pixel.color={float(u*a.col.x+v*d.col.x+w*e.col.x),float(u*a.col.y+v*d.col.y+w*e.col.y),float(u*a.col.z+v*d.col.z+w*e.col.z),1};
        }
    }
    return out;
}
double coverage_distance(const Raster& a,const Raster& b,uint8_t ss,bool scalar) {
    if(a.width!=b.width||a.height!=b.height||!ss||a.pixels.size()!=size_t(a.width)*a.height||b.pixels.size()!=a.pixels.size())throw std::invalid_argument("raster shapes differ");
    bool ca=false,cb=false,same=true;
    for(size_t i=0;i<a.pixels.size();++i){ca|=bool(a.pixels[i].covered);cb|=bool(b.pixels[i].covered);same&=a.pixels[i].covered==b.pixels[i].covered;}
    if(same)return 0;if(ca!=cb)return inf;
    double maximum=0;
    auto directed=[&](const Raster& from,const Raster& to) {
        auto field=distance_field(to,scalar);
        for(size_t i=0;i<from.pixels.size();++i)if(from.pixels[i].covered)maximum=std::max(maximum,double(field[i]));
    };
    directed(a,b);directed(b,a);return std::sqrt(maximum)/ss;
}
double attributed_distance(const Raster& a,const Raster& b,const EvalSettings& s,double limit) {
    if(s.profile==Profile::Coverage || (s.weights.normal==0 && (s.profile!=Profile::Attributes||(s.weights.color==0&&s.weights.material==0))))return 0;
    if(a.width!=b.width||a.height!=b.height||!s.supersample||!std::isfinite(limit)||limit<0||a.pixels.size()!=size_t(a.width)*a.height||b.pixels.size()!=a.pixels.size())throw std::invalid_argument("invalid attributed raster settings");
    const double invs2=1.0/(s.supersample*s.supersample),limit2=limit*limit;
    double maximum=0;
    auto directed=[&](const Raster& from,const Raster& to) {
        int radius=int(std::min(double(std::max(from.width,from.height)),std::ceil(limit*s.supersample)));
        for(int y=0;y<int(from.height);++y) {
            if(s.cancelled&&s.cancelled())return inf;
            for(int x=0;x<int(from.width);++x) {
                auto& p=from.pixels[size_t(y)*from.width+x];if(!p.visible)continue;
                double best=inf;auto& same=to.pixels[size_t(y)*to.width+x];
                if(same.visible)best=sample_cost(p,same,0,s);
                if(best<=1e-18)continue;
                int r=int(std::min(double(radius),std::ceil(std::sqrt(std::min(best,limit2))*s.supersample)));
                for(int yy=std::max(0,y-r);yy<=std::min(int(to.height)-1,y+r);++yy)
                    for(int xx=std::max(0,x-r);xx<=std::min(int(to.width)-1,x+r);++xx) {
                        auto& q=to.pixels[size_t(yy)*to.width+xx];if(!q.visible)continue;
                        double spatial=spatial_scalar(x,y,xx,yy,invs2);
                        if(spatial>=best||spatial>limit2)continue;
                        best=std::min(best,sample_cost(p,q,spatial,s));
                    }
                if(best>limit2)return inf;
                maximum=std::max(maximum,best);
            }
        }
        return maximum;
    };
    if(!std::isfinite(directed(a,b))||!std::isfinite(directed(b,a)))return inf;
    return std::sqrt(maximum);
}
Measurement evaluate(MeshView reference,MeshView candidate,const Bounds& b,const EvalSettings& s) {
    Measurement result;result.supersample=s.supersample;
    if(!(s.screen_size>0)||!std::isfinite(s.screen_size)||!(b.radius>0)||!std::isfinite(b.radius)||!finite(b.center)||unsigned(s.profile)>2
      ||!(s.limit>=0)||!std::isfinite(s.limit)||!s.supersample||s.max_supersample<s.supersample||s.max_supersample>32
      ||!std::isfinite(s.weights.normal)||!std::isfinite(s.weights.color)||!std::isfinite(s.weights.material)
      ||s.weights.normal<0||s.weights.color<0||s.weights.material<0
      ||s.weights.normal>1e12||s.weights.color>1e12||s.weights.material>1e12)throw std::invalid_argument("invalid evaluation settings");
    auto views=cameras(b,s.screen_size,s.views);
    if(views.empty())throw std::invalid_argument("at least one camera required");
    if(identical(reference,candidate))return result;
    for(uint32_t v=0;v<views.size();++v) {
        if(s.cancelled&&s.cancelled()){result.complete=false;result.passed=false;return result;}
        Measurement current;current.worst_view=v;
        for(unsigned ss=s.supersample;;ss=std::min<unsigned>(s.max_supersample,ss*2)) {
            Raster a,c;
            try {a=rasterize(reference,b,views[v],s.screen_size,uint8_t(ss),s.force_two_sided);
                 c=rasterize(candidate,b,views[v],s.screen_size,uint8_t(ss),s.force_two_sided);}
            catch(const std::length_error&) {result.complete=false;result.passed=false;result.resource_limited=true;result.error=inf;return result;}
            current.supersample=uint8_t(ss);
            current.coverage=coverage_distance(a,c,uint8_t(ss),s.force_scalar);
            current.coverage_upper=current.coverage+2*std::sqrt(2.0)/ss+1e-6;
            if(c.clipped||a.clipped)current.coverage_upper=inf;
            size_t changed=0,total=0;
            for(size_t i=0;i<a.pixels.size();++i) {
                changed+=a.pixels[i].covered!=c.pixels[i].covered;
                total+=a.pixels[i].covered||c.pixels[i].covered;
                if(s.profile!=Profile::Coverage&&a.pixels[i].visible&&c.pixels[i].visible)
                    current.normal_degrees=std::max(current.normal_degrees,std::acos(std::clamp(dot(a.pixels[i].normal,c.pixels[i].normal),-1.0,1.0))*180/pi);
            }
            current.changed_area=total?double(changed)/total:0;
            auto config=s;config.supersample=uint8_t(ss);
            current.error=current.coverage_upper>s.limit?current.coverage_upper:std::max(current.coverage_upper,attributed_distance(a,c,config,s.limit));
            current.passed=current.error<=s.limit;
            // Refine only coverage uncertainty; sampled appearance failures remain explicit.
            if(current.passed||ss>=s.max_supersample||current.coverage-2*std::sqrt(2.0)/ss>s.limit
              ||!std::isfinite(current.error)||current.coverage_upper<=s.limit)break;
        }
        ++result.views_evaluated;
        if(current.error>result.error){result.worst_view=v;result.error=current.error;result.supersample=current.supersample;}
        result.coverage=std::max(result.coverage,current.coverage);
        result.coverage_upper=std::max(result.coverage_upper,current.coverage_upper);
        result.changed_area=std::max(result.changed_area,current.changed_area);
        result.normal_degrees=std::max(result.normal_degrees,current.normal_degrees);
        if(!current.passed){result.passed=false;result.complete=false;return result;}
    }
    return result;
}
}
