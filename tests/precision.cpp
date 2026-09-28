#include "blitz/remesher.hpp"
#include "coverage.hpp"
#include <bit>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error("line "+std::to_string(__LINE__)+": " #x);}while(0)
template<class F> void throws(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}CHECK(failed);}
static Mesh grid(unsigned n,float scale=1,float bend=0) {
    Mesh m;
    for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x) {
        float u=float(x)/n,v=float(y)/n;
        m.positions.push_back({scale*(u+8),scale*(v-8),scale*bend*u*v});
        m.uv.push_back({u*4-2,v*3-1});m.normals.push_back({0,0,1});
        m.colors.push_back({17,128,249,73});
    }
    for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x) {
        uint32_t a=y*(n+1)+x,b=a+1,c=a+n+1,d=c+1;
        m.indices.insert(m.indices.end(),{a,b,d,a,d,c});
    }
    m.double_sided={1};return m;
}
static void mask_matches(MeshView mesh,const Bounds& bounds,const Camera& camera,double screen,uint8_t ss,bool two) {
    auto full=rasterize(mesh,bounds,camera,screen,ss,two);
    auto mask=detail::rasterize_coverage(mesh,bounds,camera,screen,ss,two);
    CHECK(full.width==mask.width&&full.height==mask.height&&full.clipped==mask.clipped);
    for(size_t i=0;i<full.pixels.size();++i)CHECK(bool(full.pixels[i].covered)==mask.covered(i));
    auto tail=full.pixels.size()%64;
    if(tail)CHECK((mask.words.back()>>tail)==0);
}
static double brute(const detail::CoverageRaster& a,const detail::CoverageRaster& b,uint8_t ss) {
    double worst=0;size_t count=size_t(a.width)*a.height;
    for(auto pair:{std::pair(&a,&b),std::pair(&b,&a)})for(size_t i=0;i<count;++i)if(pair.first->covered(i)) {
        double best=INFINITY;
        for(size_t j=0;j<count;++j)if(pair.second->covered(j)) {
            double x=double(i%a.width)-double(j%a.width),y=double(i/a.width)-double(j/a.width);
            best=std::min(best,x*x+y*y);
        }
        worst=std::max(worst,best);
    }
    return std::sqrt(worst)/ss;
}
int main() {
    try {
        for(unsigned k=0;k<256;++k) {
            auto c=quantize_color(k/255.,(255-k)/255.,k/255.,k/255.);
            CHECK(c.r==k&&c.g==255-k&&c.b==k&&c.a==k);
            auto f=linear_color(c);CHECK(std::abs(f.x-k/255.)<3e-8);
        }
        CHECK(quantize_color(.5,0,1).r==128);
        CHECK(quantize_color(0,1,0).a==255);
        CHECK(quantize_color(std::nextafter(.5,0.),0,0).r==127);
        for(double bad:std::array<double,5>{-.001,1.001,INFINITY,-INFINITY,std::numeric_limits<double>::quiet_NaN()}) {
            throws([&]{quantize_color(bad,0,0);});throws([&]{quantize_color(0,bad,0);});
            throws([&]{quantize_color(0,0,bad);});throws([&]{quantize_color(0,0,0,bad);});
        }
        for(float scale:{1e-12f,1.f,1e12f})for(float bend:{0.f,1e-5f,.1f}) {
            auto m=grid(20,scale,bend);auto original=m;
            for(auto mode:{OutputMode::Rebuild,OutputMode::Reuse}) {
                ReductionStats stats;ReduceSettings rs;rs.output=mode;rs.target_triangles=200;rs.statistics=&stats;
                auto reduced=reduce(m.view(),rs);auto v=reduced.view(m.view());
                CHECK(validate(v).empty());CHECK(v.triangles()<=m.view().triangles());CHECK(stats.collapsed>0);
                for(size_t i=0;i<v.colors.count;++i)CHECK(v.colors[i]==m.colors[0]);
                if(mode==OutputMode::Reuse)CHECK(reduced.shared_vertices&&v.colors.data==m.view().colors.data);
                CHECK(same_mesh_data(m.view(),original.view()));CHECK(stats.nonfinite_solves==0&&stats.nonfinite_costs==0);
            }
        }
        auto m=grid(4);auto bounds0=bounds(m.view());
        for(auto seed:{17u,9381u})for(auto ss:{uint8_t(1),uint8_t(2),uint8_t(4),uint8_t(8)}) {
            auto views=cameras(bounds0,17.25,{8,4,seed});
            for(auto c:views)for(bool two:{false,true})mask_matches(m.view(),bounds0,c,17.25,ss,two);
        }
        // Culling, zero-area faces, subpixel geometry, clipping and behind-camera vertices.
        m.double_sided={0};m.indices.insert(m.indices.end(),{0,0,0});
        auto camera=cameras(bounds0,17.25,{1,1,99}).back();
        for(auto& p:m.positions)p.z+=.00001f;
        mask_matches(m.view(),bounds0,camera,17.25,2,false);
        for(auto& p:m.positions)p.x+=100;
        mask_matches(m.view(),bounds0,camera,17.25,2,true);
        camera.distance=-100;mask_matches(m.view(),bounds0,camera,17.25,2,true);
        std::mt19937 random(2921);
        for(unsigned trial=0;trial<40;++trial) {
            detail::CoverageRaster a{13,11,std::vector<uint64_t>(3)},b=a;
            for(size_t i=0;i<143;++i){if(random()%5==0)a.words[i/64]|=uint64_t(1)<<(i%64);if(random()%7==0)b.words[i/64]|=uint64_t(1)<<(i%64);}
            for(auto ss:{uint8_t(1),uint8_t(2),uint8_t(8)}) {
                auto expected=brute(a,b,ss);
                CHECK(detail::coverage_distance(a,b,ss,true)==expected);
                CHECK(detail::coverage_distance(a,b,ss,false)==expected);
            }
        }
        detail::CoverageRaster empty{1,1,{0}},one{1,1,{1}};
        CHECK(detail::coverage_distance(empty,empty,1,true)==0);
        CHECK(std::isinf(detail::coverage_distance(empty,one,1,true)));
        auto malformed=one;malformed.words.clear();throws([&]{detail::coverage_distance(one,malformed,1,true);});
        m=grid(8);
        struct Slot {uint8_t prefix;ColorRGBA8 color;uint8_t padding[3];};
        std::vector<Slot> slots(m.colors.size());for(auto& s:slots)s={19,{17,128,249,73},{7,8,9}};
        const auto before=slots;auto view=m.view();view.colors.data=reinterpret_cast<const std::byte*>(slots.data())+offsetof(Slot,color);view.colors.stride=sizeof(Slot);
        Settings settings;settings.research.output=OutputMode::Reuse;settings.research.chain=ChainMode::Progressive;
        settings.levels=3;settings.base_pixels=16;settings.last_pixels=8;settings.profile=Profile::Attributes;
        settings.search_views={2,1,17};settings.audit_views={4,1,31};settings.search_supersample=settings.audit_supersample=2;settings.max_supersample=4;
        settings.candidate_budget=2;settings.beam_width=1;PerformanceStats work;work.raster_ns=UINT64_MAX;settings.performance=&work;
        auto result=generate(view,settings);CHECK(result.status==Status::Complete);
        for(auto& l:result.lods){CHECK(l.view(view).colors.data==view.colors.data);CHECK(l.adjacent.passed&&l.source_error.passed);}
        CHECK(std::memcmp(before.data(),slots.data(),slots.size()*sizeof(Slot))==0);
        CHECK(work.reduction_ns>0&&work.raster_ns>0&&work.raster_ns<UINT64_MAX&&work.distance_ns>0);
        CHECK(work.singular_solves<=work.solve_attempts);
        std::cout<<"RGBA8, precision robustness and packed coverage contracts passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
