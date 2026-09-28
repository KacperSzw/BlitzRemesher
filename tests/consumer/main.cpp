#include <blitz/remesher.hpp>
#include <blitz/render_cost.hpp>
int main(){
    blitz::Settings settings;if(!blitz::validate(settings).empty())return 1;
    blitz::Mesh mesh;mesh.positions={{-1,-1,0},{1,-1,0},{0,1,0}};mesh.indices={0,1,2};
    mesh.colors={{0,128,255,255},{255,0,17,255},{24,31,128,255}};
    if(sizeof(blitz::ColorRGBA8)!=4||blitz::quantize_color(0,.5,1)!=mesh.colors[0])return 4;
    blitz::Result r;r.source=mesh.view();r.lods.resize(2);
    for(auto& l:r.lods)l.data.indices=mesh.indices;
    if(blitz::runtime_levels(r)!=std::vector<uint8_t>{0})return 2;
    blitz::Camera camera{{1,0,0},{0,1,0},{0,0,1},10,1,1,false};
    auto cost=blitz::render_cost(mesh.view(),blitz::bounds(mesh.view()),camera,2);
    return cost.complete&&cost.submitted==1&&cost.covered_samples>0?0:3;
}
