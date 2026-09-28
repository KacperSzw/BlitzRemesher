#include "blitz/remesher.hpp"
#include "coverage.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
using namespace blitz;
using Clock=std::chrono::steady_clock;
using json=nlohmann::json;
static json samples(std::vector<double> times) {
    auto raw=times;std::sort(times.begin(),times.end());
    return {{"samples_ms",raw},{"median_ms",times[times.size()/2]},{"min_ms",times.front()},{"max_ms",times.back()}};
}
int main() {
    try {
        Mesh mesh;constexpr unsigned n=96;
        for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x) {
            float u=float(x)/n,v=float(y)/n;
            mesh.positions.push_back({u,v,.03f*std::sin(u*13)*std::cos(v*9)});
            mesh.colors.push_back(quantize_color(u,v,.5));mesh.normals.push_back({0,0,1});mesh.uv.push_back({u*2,v*3});
        }
        for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){uint32_t a=y*(n+1)+x,b=a+1,c=a+n+1,d=c+1;mesh.indices.insert(mesh.indices.end(),{a,b,d,a,d,c});}
        mesh.double_sided={1};auto moved=mesh;for(auto& p:moved.positions)p.x+=.005f;
        auto b=bounds(mesh.view());auto camera=cameras(b,512,{1,0,871})[0];
        auto storage=reduction_storage();json report={{"purpose","Controlled CPU kernel timings; procedural fixture, no visual SCORE"},
          {"compiler",__VERSION__},{"quadric_bytes",storage.quadric_bytes},{"candidate_bytes",storage.candidate_bytes},
          {"mesh_vertices",mesh.positions.size()},{"mesh_triangles",mesh.view().triangles()},{"repetitions",7},{"warmups",1},{"screen_pixels",512},{"supersample",4}};
        std::array<std::vector<double>,2> raster_times,distance_times;
        double checksum=0;
        for(int trial=-1;trial<7;++trial)for(int order=0;order<2;++order) {
            int packed=(trial+1+order)%2;auto start=Clock::now();double distance;
            if(packed) {
                auto a=detail::rasterize_coverage(mesh.view(),b,camera,512,4,false),c=detail::rasterize_coverage(moved.view(),b,camera,512,4,false);
                auto rendered=Clock::now();distance=detail::coverage_distance(a,c,4,false);auto done=Clock::now();
                if(trial>=0){raster_times[1].push_back(std::chrono::duration<double,std::milli>(rendered-start).count());distance_times[1].push_back(std::chrono::duration<double,std::milli>(done-rendered).count());}
            } else {
                auto a=rasterize(mesh.view(),b,camera,512,4,false),c=rasterize(moved.view(),b,camera,512,4,false);
                auto rendered=Clock::now();distance=coverage_distance(a,c,4,false);auto done=Clock::now();
                if(trial>=0){raster_times[0].push_back(std::chrono::duration<double,std::milli>(rendered-start).count());distance_times[0].push_back(std::chrono::duration<double,std::milli>(done-rendered).count());}
            }
            if(!std::isfinite(distance))throw std::runtime_error("nonfinite microbenchmark distance");
            if(checksum!=0&&distance!=checksum)throw std::runtime_error("packed/full raster distance mismatch");checksum=distance;
        }
        report["distance_px"]=checksum;
        for(int i=0;i<2;++i)report[i?"packed":"full"]={{"raster",samples(raster_times[i])},{"distance",samples(distance_times[i])}};
        std::vector<double> reduction_times;uint64_t output_hash=0;ReductionStats stats;
        for(int trial=-1;trial<7;++trial) {
            ReduceSettings rs;rs.target_triangles=2048;rs.statistics=&stats;rs.normal_weight=1;
            auto start=Clock::now();auto lod=reduce(mesh.view(),rs);
            double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();if(trial>=0)reduction_times.push_back(elapsed);
            uint64_t hash=14695981039346656037ull;
            auto add=[&](auto span){for(auto value:std::as_bytes(span)){hash^=std::to_integer<uint8_t>(value);hash*=1099511628211ull;}};
            add(std::span(lod.data.positions));add(std::span(lod.data.indices));add(std::span(lod.data.colors));
            if(output_hash&&hash!=output_hash)throw std::runtime_error("nondeterministic reduction");output_hash=hash;
            report["output_triangles"]=lod.view(mesh.view()).triangles();
        }
        report["reduction"]=samples(reduction_times);report["output_fnv1a64"]=output_hash;
        report["numerics"]={{"solve_attempts",stats.solve_attempts},{"singular_solves",stats.singular_solves},{"nonfinite_solves",stats.nonfinite_solves},{"position_fallbacks",stats.position_fallbacks},{"nonfinite_costs",stats.nonfinite_costs}};
        std::cout<<report.dump(2)<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
