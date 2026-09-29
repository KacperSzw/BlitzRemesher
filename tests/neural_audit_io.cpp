#include "neural_audit_io.hpp"
#include <iostream>
using namespace blitz;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    if(!neural_available())return 77;
    Mesh m;m.positions={{-1,-1,0},{1,-1,0},{0,1,0}};m.normals={{0,0,1},{0,0,1},{0,0,1}};m.indices={0,1,2};m.materials={0};m.double_sided={1};m.colors={{0,255,64,123},{32,0,128,0},{255,0,0,255}};
    check(same_mesh_data(m.view(),audit_mesh(audit_mesh(m)).view()),"audit replay changed mesh streams");
    auto dir=std::filesystem::temp_directory_path()/"blitz-audit-replay-contract";std::filesystem::create_directories(dir);auto file=dir/"replay.json";
    struct Cleanup{std::filesystem::path dir;~Cleanup(){std::filesystem::remove_all(dir);}} cleanup{dir};
    NeuralStats stats;auto& failure=stats.confirmation_failure.emplace();failure.reference=m;failure.candidate=m;failure.candidate.normals[1]=normalized({.2f,0,1});failure.bounds=bounds(m.view());
    failure.settings.screen_size=16;failure.settings.views={2,1,713};failure.settings.supersample=2;failure.settings.max_supersample=4;failure.settings.limit=3;
    save_audit_failure(stats,file);NeuralOptions options;options.memory_mib=256;
    auto result=replay_audit(file,options);check(result.at("decisions_agree").get<bool>(),"serialized candidate replay changed decisions");
    nlohmann::json j;{std::ifstream in(file);in>>j;}j["candidate"]["indices"][0]=1;{std::ofstream out(file);out<<j;}
    bool rejected=false;try{replay_audit(file,options);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"replay accepted corrupted candidate");
    std::cout<<"audit replay contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
