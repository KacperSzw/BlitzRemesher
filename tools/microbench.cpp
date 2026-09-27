#include "blitz/evaluate.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
int main() {
    using namespace blitz;using clock=std::chrono::steady_clock;
    constexpr unsigned width=640;
    Raster a{width,width,std::vector<Pixel>(width*width)},b=a;
    for(unsigned y=0;y<width;++y)for(unsigned x=0;x<width;++x) {
        double dx=int(x)-320,dy=int(y)-320;auto i=y*width+x;
        a.pixels[i].covered=dx*dx+dy*dy<230*230;
        b.pixels[i].covered=(dx-5)*(dx-5)+(dy+2)*(dy+2)<225*225;
    }
    nlohmann::json report={{"fixture","640px filled discs; coverage EDT"},{"compiler",__VERSION__},{"backend",evaluator_backend()},{"repetitions",21}};
    for(bool scalar:{true,false}) {
        std::vector<double> times;double checksum=0;
        for(int trial=0;trial<24;++trial){auto start=clock::now();auto d=coverage_distance(a,b,8,scalar);
            double elapsed=std::chrono::duration<double,std::milli>(clock::now()-start).count();checksum+=d;if(trial>=3)times.push_back(elapsed);}
        std::sort(times.begin(),times.end());report[scalar?"scalar":"dispatch"]={{"median_ms",times[times.size()/2]},{"min_ms",times.front()},{"max_ms",times.back()},{"checksum",checksum}};
    }
    report["speedup"]=report["scalar"]["median_ms"].get<double>()/report["dispatch"]["median_ms"].get<double>();
    std::cout<<report.dump(2)<<'\n';
}
