#include "blitz/render_cost.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error("render-cost contract: " #x);}while(0)
int main(){
    try {
        Mesh m;m.positions={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};m.indices={0,1,2,0,2,3};
        Bounds b{{0,0,0},2};Camera camera{{1,0,0},{0,1,0},{0,0,1},10,1,1,false};
        // A 2x2 square exactly fills one quad. The diagonal belongs to one
        // triangle only; each primitive still requires its own quad.
        auto cost=render_cost(m.view(),b,camera,2);
        CHECK(cost.complete&&cost.views==1&&cost.submitted==2);
        CHECK(cost.covered_samples==4&&cost.covered_pixels==4&&cost.primitive_quads==2);
        CHECK(cost.under_one_px2==0&&cost.under_four_px2==2&&cost.zero_samples==0);
        m.indices.insert(m.indices.end(),{0,1,2,0,2,3});cost=render_cost(m.view(),b,camera,2);
        CHECK(cost.covered_samples==8&&cost.covered_pixels==4&&cost.primitive_quads==4);
        m.indices={2,1,0};cost=render_cost(m.view(),b,camera,2);CHECK(cost.culled==1&&cost.covered_samples==0);
        m.double_sided={1};cost=render_cost(m.view(),b,camera,2);CHECK(cost.culled==0&&cost.covered_samples==3);
        m.positions={{.1f,.1f,0},{.9f,.1f,0},{.5f,.9f,0}};m.indices={0,1,2};
        cost=render_cost(m.view(),b,camera,2);CHECK(cost.under_one_px2==1&&cost.covered_samples==1&&cost.primitive_quads==1);
        m.positions={{-.4f,-.4f,0},{.4f,-.4f,0},{-.4f,.4f,0}};
        cost=render_cost(m.view(),b,camera,2);CHECK(cost.zero_samples==1&&cost.covered_samples==0&&cost.primitive_quads==0);
        m.positions[0].x=-100;cost=render_cost(m.view(),b,camera,2);CHECK(cost.clipped&&!cost.complete);
        std::cout<<"Quad occupancy, shared edges, overlap, culling and tiny-triangle contracts passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
