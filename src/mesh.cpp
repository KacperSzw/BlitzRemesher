#include "blitz/mesh.hpp"
#include <limits>
#include <stdexcept>
namespace blitz {
ColorRGBA8 quantize_color(double r,double g,double b,double a) {
    auto channel=[](double v) {
        if(!std::isfinite(v)||v<0||v>1)throw std::invalid_argument("color channels must be finite and in [0,1]");
        return uint8_t(std::floor(v*255+.5));
    };
    return {channel(r),channel(g),channel(b),channel(a)};
}
Bounds bounds(MeshView m) {
    if(!m.positions) return {};
    Vec3 lo=m.positions[0],hi=lo;
    for(size_t i=1;i<m.positions.count;++i) {auto p=m.positions[i];
      lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
      hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};}
    Bounds b{{float((double(lo.x)+hi.x)*.5),float((double(lo.y)+hi.y)*.5),float((double(lo.z)+hi.z)*.5)},0};
    for(size_t i=0;i<m.positions.count;++i) {auto p=m.positions[i];double x=double(p.x)-b.center.x,y=double(p.y)-b.center.y,z=double(p.z)-b.center.z;b.radius=std::max(b.radius,std::sqrt(x*x+y*y+z*z));}
    return b;
}
template<class T> static bool valid_stream(Stream<T> s,size_t n,bool required=false) {
    if(!s.count)return !required;
    if(!s.data) return !required && !s.count;
    return s.count==n&&s.stride>=sizeof(T)&&(!n||(n-1)<=(SIZE_MAX-sizeof(T))/s.stride);
}
std::string validate(MeshView m) {
    size_t n=m.positions.count;
    if(!n||n>=UINT32_MAX||m.indices.empty()||m.indices.size()%3||m.indices.size()>=UINT32_MAX) return "mesh requires nonempty triangle indices and 32-bit representable counts";
    if(!valid_stream(m.positions,n,true)||!valid_stream(m.normals,n)||!valid_stream(m.uv,n)||!valid_stream(m.colors,n)||!valid_stream(m.tangents,n)) return "invalid stream count, pointer or stride";
    if(!m.materials.empty()&&m.materials.size()!=m.triangles()) return "material count differs from triangle count";
    for(auto i:m.indices) if(i>=n)return "index outside vertex stream";
    for(size_t i=0;i<n;++i) {
        if(!finite(m.positions[i])||(m.normals&&!finite(m.normals[i]))) return "nonfinite position or normal";
        if(m.uv) {auto t=m.uv[i];if(!std::isfinite(t.x)||!std::isfinite(t.y))return "nonfinite UV";}
        if(m.tangents) {auto t=m.tangents[i];if(!std::isfinite(t.x)||!std::isfinite(t.y)||!std::isfinite(t.z)||!std::isfinite(t.w))return "nonfinite attribute";}
    }
    auto b=bounds(m); if(!(b.radius>0)||!std::isfinite(b.radius)) return "zero or invalid source extent";
    return {};
}
template<class T> static std::vector<T> copy(Stream<T> s) {std::vector<T> v(s.count);for(size_t i=0;i<s.count;++i)v[i]=s[i];return v;}
Mesh copy_mesh(MeshView v) {
    return {copy(v.positions),copy(v.normals),copy(v.uv),copy(v.colors),copy(v.tangents),
      {v.indices.begin(),v.indices.end()},{v.materials.begin(),v.materials.end()},{v.double_sided.begin(),v.double_sided.end()}};
}
template<class T> static bool same_stream(Stream<T> a,Stream<T> b) {
    if(a.count!=b.count)return false;
    if(!a.count||(a.data==b.data&&a.stride==b.stride))return true;
    if(a.stride==sizeof(T)&&b.stride==sizeof(T))return std::memcmp(a.data,b.data,a.count*sizeof(T))==0;
    for(size_t i=0;i<a.count;++i)if(std::memcmp(a.data+i*a.stride,b.data+i*b.stride,sizeof(T)))return false;
    return true;
}
template<class T> static bool same_span(std::span<const T> a,std::span<const T> b) {
    return a.size()==b.size()&&(a.empty()||a.data()==b.data()||std::memcmp(a.data(),b.data(),a.size_bytes())==0);
}
bool same_mesh_data(MeshView a,MeshView b) {
    return same_span(a.indices,b.indices)&&same_span(a.materials,b.materials)&&same_span(a.double_sided,b.double_sided)
      &&same_stream(a.positions,b.positions)&&same_stream(a.normals,b.normals)&&same_stream(a.uv,b.uv)
      &&same_stream(a.colors,b.colors)&&same_stream(a.tangents,b.tangents);
}
void compact(Mesh& m) {
    std::vector<uint32_t> map(m.positions.size(),UINT32_MAX);
    Mesh out;out.materials=std::move(m.materials);out.double_sided=std::move(m.double_sided);
    out.indices.reserve(m.indices.size());
    for(auto i:m.indices) {
        if(map[i]==UINT32_MAX) {
            map[i]=uint32_t(out.positions.size());out.positions.push_back(m.positions[i]);
            if(!m.normals.empty())out.normals.push_back(m.normals[i]);
            if(!m.uv.empty())out.uv.push_back(m.uv[i]);
            if(!m.colors.empty())out.colors.push_back(m.colors[i]);
            if(!m.tangents.empty())out.tangents.push_back(m.tangents[i]);
        }
        out.indices.push_back(map[i]);
    }
    m=std::move(out);
}
Distortion uv_distortion(MeshView m) {
    Distortion d;if(!m.uv)return d;double sum=0;uint64_t count=0;double scale=bounds(m).diameter();
    for(size_t f=0;f<m.triangles();++f) {
        auto a=m.indices[f*3],b=m.indices[f*3+1],c=m.indices[f*3+2];
        auto x=m.uv[a],y=m.uv[b],z=m.uv[c];
        double ua=(double(y.x)-x.x)*(double(z.y)-x.y)-(double(y.y)-x.y)*(double(z.x)-x.x);
        auto edge=[&](uint32_t i){auto p=m.positions[i],p0=m.positions[a];return Vec3{float((double(p.x)-p0.x)/scale),float((double(p.y)-p0.y)/scale),float((double(p.z)-p0.z)/scale)};};
        auto e0=edge(b),e1=edge(c);double pa=length(cross(e0,e1));
        if(ua<0)++d.negative_uv_faces;
        if(std::abs(ua)<1e-20)++d.degenerate_uv_faces;
        if(pa>0) {
            double density=std::sqrt(std::abs(ua)/pa)/scale;sum+=density;d.max_uv_density=std::max(d.max_uv_density,density);++count;
            double l=length(e0),ex=dot(e0,e1)/l,ey=pa/l;
            double a00=(double(y.x)-x.x)/l,a10=(double(y.y)-x.y)/l;
            double a01=(double(z.x)-x.x-a00*ex)/ey,a11=(double(z.y)-x.y-a10*ex)/ey;
            double trace=a00*a00+a01*a01+a10*a10+a11*a11,det=a00*a11-a01*a10;
            double largest=.5*(trace+std::sqrt(std::max(0.0,trace*trace-4*det*det)));
            double ratio=std::abs(det)>1e-30?largest/std::abs(det):std::numeric_limits<double>::infinity();
            d.max_uv_anisotropy=std::max(d.max_uv_anisotropy,ratio);
        }
    }
    d.mean_uv_density=count?sum/count:0;return d;
}
}
