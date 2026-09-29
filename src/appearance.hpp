#pragma once
// Appearance fields use scaled normals/RGB and a common area coefficient.
// Per wedge: 10 positional coefficients, area, then four coefficients/channel.
// Allocation is contiguous and proportional only to present, active channels.
#include "blitz/remesher.hpp"
#include <numeric>
namespace blitz::detail {
class Appearance {
    static constexpr uint32_t none=UINT32_MAX;
    uint8_t channels{},stride{};
    bool normals{},colors{},coupled{};
    double nw{},cw{};
    const ReduceSettings& settings;
    std::vector<double> coefficients;
    std::vector<uint32_t> head,tail,next;
    std::vector<Vec3> normal;
    std::vector<ColorRGBA8> color;
    using Block=std::array<double,35>;
    const double* block(uint32_t i) const {return coefficients.data()+size_t(i)*stride;}
    double* block(uint32_t i) {return coefficients.data()+size_t(i)*stride;}
    static void plane(double* q,const double* v,double w) {
        unsigned k=0;for(unsigned i=0;i<4;++i)for(unsigned j=i;j<4;++j)q[k++]+=v[i]*v[j]*w;
    }
    static double positional(const double* q,Vec3 p) {
        const double v[4]={p.x,p.y,p.z,1};double sum=0;unsigned k=0;
        for(unsigned i=0;i<4;++i)for(unsigned j=i;j<4;++j)sum+=q[k++]*v[i]*v[j]*(i==j?1:2);return sum;
    }
    void values(uint32_t i,double* out) const {
        unsigned k=0;if(normals){auto n=normalized(normal[i]);out[k++]=n.x*nw;out[k++]=n.y*nw;out[k++]=n.z*nw;}
        if(colors){auto c=color[i];out[k++]=double(c.r)/255*cw;out[k++]=double(c.g)/255*cw;out[k++]=double(c.b)/255*cw;}
    }
    void stored(double* a,uint32_t fallback) const {
        unsigned k=0;if(normals){Vec3 n{float(a[0]/nw),float(a[1]/nw),float(a[2]/nw)};
            n=length(n)>1e-20&&finite(n)?normalized(n):normalized(normal[fallback]);
            a[k++]=n.x*nw;a[k++]=n.y*nw;a[k++]=n.z*nw;}
        if(colors)for(unsigned j=0;j<3;++j,++k)a[k]=std::floor(std::clamp(a[k]/cw,0.,1.)*255+.5)/255*cw;
    }
    void emitted(const double* q,uint32_t u,uint32_t v,Vec3 point,Vec3 pu,Vec3 pv,double* a) const {
        values(u,a);
        if(settings.output==OutputMode::Reuse)return;
        if(settings.appearance_stage>=AppearanceStage::Attributes&&q[10]>0) {
            for(unsigned j=0;j<channels;++j){auto b=q+11+j*4;a[j]=(b[0]*point.x+b[1]*point.y+b[2]*point.z+b[3])/q[10];}
        } else if(!coupled) {
            double z[6];values(v,z);auto edge=pv-pu;auto len=dot(edge,edge);auto t=len?std::clamp(dot(point-pu,edge)/len,0.,1.):.5;
            for(unsigned j=0;j<channels;++j)a[j]=a[j]*(1-t)+z[j]*t;
            if(normals){auto n=normal[u]*(1-t)+normal[v]*t;a[0]=n.x*nw;a[1]=n.y*nw;a[2]=n.z*nw;}
        }
        stored(a,u);
    }
    template<class F> void groups(uint32_t u,uint32_t v,F f) const {
        if(!coupled){Block q{};for(unsigned j=0;j<stride;++j)q[j]=block(u)[j]+block(v)[j];f(q.data(),u,v);}
        else for(auto h:{head[u],head[v]})for(auto i=h;i!=none;i=next[i])f(block(i),i,i);
    }
public:
    Appearance(MeshView source,const ReduceSettings& s,std::span<const uint32_t> map={},size_t positions=0):settings(s) {
        if(s.appearance_stage==AppearanceStage::Off)return;
        normals=bool(source.normals)&&s.appearance_weights.normal>0;colors=bool(source.colors)&&s.appearance_weights.color>0;
        channels=uint8_t(3*normals+3*colors);if(!channels)return;
        stride=uint8_t(11+4*channels);nw=s.appearance_weights.normal;cw=s.appearance_weights.color;coupled=!map.empty();
        const auto n=source.positions.count;coefficients.resize(n*stride);
        if(normals){normal.resize(n);for(size_t i=0;i<n;++i)normal[i]=source.normals[i];}
        if(colors){color.resize(n);for(size_t i=0;i<n;++i)color[i]=source.colors[i];}
        if(coupled){head.assign(positions,none);tail=head;next.assign(n,none);
            for(uint32_t i=0;i<n;++i){auto p=map[i];if(head[p]==none)head[p]=i;else next[tail[p]]=i;tail[p]=i;}}
        auto bounds_=bounds(source);double scale=bounds_.diameter();
        for(size_t f=0;f<source.triangles();++f) {
            uint32_t ids[3]={source.indices[3*f],source.indices[3*f+1],source.indices[3*f+2]};
            double p[3][3],a[3][6];for(unsigned i=0;i<3;++i){auto x=source.positions[ids[i]];
                p[i][0]=(double(x.x)-bounds_.center.x)/scale;p[i][1]=(double(x.y)-bounds_.center.y)/scale;p[i][2]=(double(x.z)-bounds_.center.z)/scale;values(ids[i],a[i]);}
            double e[3],z[3];for(unsigned j=0;j<3;++j){e[j]=p[1][j]-p[0][j];z[j]=p[2][j]-p[0][j];}
            double ee=0,ez=0,zz=0;for(unsigned j=0;j<3;++j){ee+=e[j]*e[j];ez+=e[j]*z[j];zz+=z[j]*z[j];}
            double det=ee*zz-ez*ez;if(!(det>1e-14*ee*zz)||det<=1e-48)continue;
            double area=std::sqrt(det);Block q{};q[10]=area;
            for(unsigned c=0;c<channels;++c){double d1=a[1][c]-a[0][c],d2=a[2][c]-a[0][c];
                double u=(d1*zz-d2*ez)/det,v=(d2*ee-d1*ez)/det,g[4];g[3]=a[0][c];
                for(unsigned j=0;j<3;++j){g[j]=u*e[j]+v*z[j];g[3]-=g[j]*p[0][j];}
                plane(q.data(),g,area);for(unsigned j=0;j<4;++j)q[11+4*c+j]=g[j]*area;}
            for(auto i:ids)for(unsigned j=0;j<stride;++j)block(i)[j]+=q[j];
        }
    }
    uint64_t bytes() const {return coefficients.capacity()*sizeof(double)+(head.capacity()+tail.capacity()+next.capacity())*sizeof(uint32_t)+normal.capacity()*sizeof(Vec3)+color.capacity()*sizeof(ColorRGBA8);}
    bool active() const {return channels!=0;}
    double cost(uint32_t u,uint32_t v,Vec3 p,Vec3 pu,Vec3 pv) const {
        if(!active())return 0;double result=0;
        groups(u,v,[&](const double* q,uint32_t i,uint32_t j){double a[6];emitted(q,i,j,p,pu,pv,a);double cost=positional(q,p);
            for(unsigned c=0;c<channels;++c){auto b=q+11+4*c;cost+=q[10]*a[c]*a[c]-2*a[c]*(b[0]*p.x+b[1]*p.y+b[2]*p.z+b[3]);}
            result+=std::max(0.,cost);});return result;
    }
    template<class Q> void add_reduced(Q& sum,uint32_t u,uint32_t v) const {
        groups(u,v,[&](const double* q,uint32_t,uint32_t){for(unsigned j=0;j<10;++j)sum.a[j]+=q[j];
            if(q[10]>0)for(unsigned c=0;c<channels;++c)plane(sum.a,q+11+c*4,-1/q[10]);});
    }
    void collapse(uint32_t u,uint32_t v,Vec3 p,Vec3 pu,Vec3 pv) {
        if(!active())return;
        groups(u,v,[&](const double* q,uint32_t i,uint32_t j){double a[6];emitted(q,i,j,p,pu,pv,a);unsigned k=0;
            if(normals){normal[i]={float(a[0]/nw),float(a[1]/nw),float(a[2]/nw)};k=3;}
            if(colors){auto c=color[i];color[i]={uint8_t(std::lround(a[k]/cw*255)),uint8_t(std::lround(a[k+1]/cw*255)),uint8_t(std::lround(a[k+2]/cw*255)),c.a};}});
        if(!coupled)for(unsigned j=0;j<stride;++j)block(u)[j]+=block(v)[j];
        else {if(head[u]==none)head[u]=head[v];else next[tail[u]]=head[v];if(tail[v]!=none)tail[u]=tail[v];head[v]=tail[v]=none;}
    }
    void write(Mesh& mesh) const {
        if(!active()||settings.output==OutputMode::Reuse||settings.appearance_stage<AppearanceStage::Attributes)return;
        if(normals)mesh.normals=normal;if(colors)mesh.colors=color;
        if(normals)for(size_t i=0;i<mesh.tangents.size();++i){auto& t=mesh.tangents[i];auto n=normal[i];Vec3 v{t.x,t.y,t.z};v=normalized(v-n*dot(v,n));
            if(length(v)<.5){auto axis=std::abs(n.x)<.8?Vec3{1,0,0}:Vec3{0,1,0};v=normalized(cross(n,axis));}t.x=v.x;t.y=v.y;t.z=v.z;}
    }
};
}
