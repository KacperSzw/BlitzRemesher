#include "blitz/render_cost.hpp"
#include <bit>
#include <stdexcept>
namespace blitz {
namespace {
struct Point { double x,y,z; };
double edge(Point a,Point b,double x,double y){return (b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);}
// Screen Y points upward. Each shared edge belongs to exactly one triangle.
bool inclusive(Point a,Point b){return b.y<a.y||(b.y==a.y&&b.x<a.x);}
bool inside(double value,bool tie){return value>0||(value==0&&tie);}
}
RenderCost render_cost(MeshView mesh,const Bounds& reference,const Camera& camera,double screen) {
    if(!std::isfinite(screen)||screen<=0||screen>16384||!std::isfinite(reference.radius)||reference.radius<=0)
        throw std::invalid_argument("invalid render-cost extent or source bounds");
    const uint32_t size=(uint32_t(std::ceil(screen+8))+1)&~1u; // Fixed even viewport origin and quad alignment.
    const size_t pixels=size_t(size)*size;
    if(pixels>64000000)throw std::length_error("render-cost viewport exceeds memory budget");
    RenderCost out;out.views=1;out.submitted=mesh.triangles();
    std::vector<uint64_t> occupied((pixels+63)/64);
    std::vector<Point> projected(mesh.positions.count);
    for(size_t i=0;i<mesh.positions.count;++i) {
        const auto p=mesh.positions[i];
        const double x=double(p.x)-reference.center.x,y=double(p.y)-reference.center.y,z=double(p.z)-reference.center.z;
        auto dot_axis=[&](Vec3 a){return x*a.x+y*a.y+z*a.z;};
        const double depth=camera.distance-dot_axis(camera.forward);
        const double scale=camera.perspective?camera.focal/depth:camera.scale;
        projected[i]={dot_axis(camera.right)*scale+size*.5,dot_axis(camera.up)*scale+size*.5,depth};
    }
    for(size_t face=0;face<mesh.triangles();++face) {
        Point a=projected[mesh.indices[3*face]],b=projected[mesh.indices[3*face+1]],c=projected[mesh.indices[3*face+2]];
        if(camera.perspective&&(a.z<=0||b.z<=0||c.z<=0)){out.clipped=true;out.complete=false;continue;}
        double area=edge(a,b,c.x,c.y);
        if(!std::isfinite(area)){out.clipped=true;out.complete=false;continue;}
        if(area==0){++out.degenerate;continue;}
        if(area<0) {
            if(!mesh.two_sided(face)){++out.culled;continue;}
            std::swap(b,c);area=-area;
        }
        ++out.projected;out.under_one_px2+=area<2;out.under_four_px2+=area<8;
        const double minx=std::min({a.x,b.x,c.x}),maxx=std::max({a.x,b.x,c.x});
        const double miny=std::min({a.y,b.y,c.y}),maxy=std::max({a.y,b.y,c.y});
        if(minx<0||miny<0||maxx>size||maxy>size){out.clipped=true;out.complete=false;continue;}
        const int x0=std::max(0,int(std::ceil(minx-.5))),x1=std::min(int(size)-1,int(std::floor(maxx-.5)));
        const int y0=std::max(0,int(std::ceil(miny-.5))),y1=std::min(int(size)-1,int(std::floor(maxy-.5)));
        const bool ab=inclusive(a,b),bc=inclusive(b,c),ca=inclusive(c,a);
        uint64_t samples=0;
        for(int y=y0&~1;y<=y1;y+=2)for(int x=x0&~1;x<=x1;x+=2) {
            unsigned mask=0;
            for(int dy=0;dy!=2;++dy)for(int dx=0;dx!=2;++dx) {
                const int xx=x+dx,yy=y+dy;
                if(xx<x0||xx>x1||yy<y0||yy>y1)continue;
                const double px=xx+.5,py=yy+.5;
                if(inside(edge(a,b,px,py),ab)&&inside(edge(b,c,px,py),bc)&&inside(edge(c,a,px,py),ca)) {
                    mask|=1u<<(2*dy+dx);const size_t pixel=size_t(yy)*size+xx;
                    occupied[pixel/64]|=uint64_t(1)<<(pixel%64);
                }
            }
            if(mask){++out.primitive_quads;samples+=std::popcount(mask);}
        }
        out.covered_samples+=samples;out.zero_samples+=samples==0;
    }
    for(auto word:occupied)out.covered_pixels+=std::popcount(word);
    return out;
}
RenderCost render_cost(MeshView mesh,const Bounds& reference,ViewSet views,double screen) {
    RenderCost out;
    for(auto camera:cameras(reference,screen,views)) {
        const auto v=render_cost(mesh,reference,camera,screen);
        out.submitted+=v.submitted;out.culled+=v.culled;out.degenerate+=v.degenerate;out.projected+=v.projected;
        out.under_one_px2+=v.under_one_px2;out.under_four_px2+=v.under_four_px2;out.zero_samples+=v.zero_samples;
        out.covered_samples+=v.covered_samples;out.primitive_quads+=v.primitive_quads;out.covered_pixels+=v.covered_pixels;
        out.views+=v.views;out.complete&=v.complete;out.clipped|=v.clipped;
    }
    return out;
}
}
