#include "blitz/io.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
using namespace blitz;using json=nlohmann::json;
int main(int argc,char** argv){try{
 if(argc!=4)throw std::invalid_argument("blitz-appearance-replay MESH CONFIG OUTPUT.json");
 std::ifstream in(argv[2]);json cfg;in>>cfg;auto s=settings_json(cfg,true);auto mesh=load_mesh(argv[1]);auto ref=bounds(mesh.view());
 json rows=json::array();const auto begin=std::chrono::steady_clock::now();
 for(double fraction:{.99,.9,.5})for(double pixels:{16.,32.}) {
  ReduceSettings rs;rs.output=OutputMode::Rebuild;rs.target_triangles=std::max<size_t>(1,size_t(mesh.view().triangles()*fraction));
  rs.coupled_wedges=s.coupled_wedges;rs.prune=false;
  auto lod=reduce(mesh.view(),rs);auto v=lod.view(mesh.view());
  // Persist exact candidate and source once; diagnostics never regenerate it.
  auto candidate=std::filesystem::path(argv[3]);candidate+="."+std::to_string(int(fraction*100))+".ply";
  if(pixels==16)save_ply(v,candidate);
  for(unsigned set=0;set<2;++set)for(uint8_t ss:{2,4,8,16,32}) {
   if(std::chrono::steady_clock::now()-begin>std::chrono::minutes(45))throw std::runtime_error("replay deadline; partial rows retained");
   EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.views=set?s.audit_views:s.search_views;e.screen_size=pixels;e.limit=2;
   e.supersample=e.max_supersample=ss;e.max_changed_area=.5;EvaluationWitness w;e.witness=&w;
   e.cancelled=[&]{return std::chrono::steady_clock::now()-begin>std::chrono::minutes(45);};
   auto t=std::chrono::steady_clock::now();auto m=evaluate(mesh.view(),v,ref,e);
   rows.push_back({{"fraction",fraction},{"triangles",v.triangles()},{"pixels",pixels},{"view_set",set},{"supersample",ss},
    {"passed",m.passed},{"complete",m.complete},{"resource_limited",m.resource_limited},{"error",m.error},{"coverage",m.coverage},{"area",m.changed_area},
    {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()},
    {"witness",{{"present",bool(w.present)},{"view",w.view},{"x",w.x},{"y",w.y},{"direction",unsigned(w.direction)},{"target_visible",bool(w.target_visible)},
     {"spatial_squared",w.spatial},{"normal_squared",w.normal},{"color_squared",w.color},{"material_squared",w.material}}}});
   std::ofstream out(argv[3]);out<<json{{"complete",false},{"input",argv[1]},{"config",cfg},{"rows",rows}}.dump(2)<<'\n';
  }
 }
 std::ofstream out(argv[3]);out<<json{{"complete",true},{"input",argv[1]},{"config",cfg},{"rows",rows}}.dump(2)<<'\n';
 std::cout<<"replay complete: "<<rows.size()<<" comparisons\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
