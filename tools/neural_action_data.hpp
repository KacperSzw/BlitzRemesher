#pragma once
#include "neural_data.hpp"
#include "neural_placement.hpp"
namespace blitz::neural::training {
constexpr uint8_t SourcePass=1,AdjacentPass=2,Preferred=4,Queried=8;
constexpr uint32_t action_pool=16;
struct ActionData {
    uint32_t architecture{action_schema};
    std::vector<float> x,progress,targets;
    std::vector<uint8_t> labels;
    std::vector<uint32_t> offsets{0},from,to;
    size_t states() const {return offsets.size()-1;}
};
inline void validate_actions(const ActionData& a) {
    auto width=policy_inputs(a.architecture);if(!width||a.offsets.empty()||a.offsets.front()!=0||a.offsets.back()!=a.labels.size()||!std::is_sorted(a.offsets.begin(),a.offsets.end())||a.x.size()!=a.labels.size()*width||a.from.size()!=a.labels.size()||a.to.size()!=a.labels.size()||a.progress.size()!=a.states()||a.targets.size()!=(a.architecture==placement_schema?a.labels.size()*9:0))throw std::invalid_argument("action shard dimensions");
    for(size_t i=0;i<a.states();++i)if(a.offsets[i+1]-a.offsets[i]<1||a.offsets[i+1]-a.offsets[i]>action_pool||!std::isfinite(a.progress[i])||a.progress[i]<0||a.progress[i]>1)throw std::invalid_argument("action state dimensions/progress");
    for(float x:a.x)if(!std::isfinite(x))throw std::invalid_argument("nonfinite action feature");
    for(float x:a.targets)if(!std::isfinite(x))throw std::invalid_argument("nonfinite placement target");
    for(uint8_t x:a.labels){if(a.architecture==action_schema){if(x>15||!(x&Queried)||((x&Preferred)&&((x&3)!=3)))throw std::invalid_argument("invalid action supervision mask");}
        else if(((x&1)&&!(x&SourceKnown))||((x&2)&&!(x&AdjacentKnown))||((x&(Preferred|PositionKnown))&&((x&27)!=27))||((x&(Normal0Known|Normal1Known))&&!(x&PositionKnown)))throw std::invalid_argument("invalid placement supervision mask");}
    if(a.architecture==placement_schema)for(size_t i=0;i<a.labels.size();++i)for(unsigned j=0;j<9;++j){float x=a.targets[i*9+j];bool known=a.labels[i]&(32u<<(j/3));if((!known&&x!=0)||(known&&std::abs(x)>(j<3?1.f:2.f)))throw std::invalid_argument("placement target outside contract");}
}
inline void save_actions(const fs::path& path,const ActionData& a) {
    static_assert(std::endian::native==std::endian::little);validate_actions(a);auto temp=path;temp+=".part";
    {std::ofstream f(temp,std::ios::binary);f.write(a.architecture==action_schema?"BLZACT02":"BLZACT03",8);write_vector(f,a.x);write_vector(f,a.labels);write_vector(f,a.offsets);write_vector(f,a.progress);write_vector(f,a.from);write_vector(f,a.to);if(a.architecture==placement_schema)write_vector(f,a.targets);f.close();if(!f)throw std::runtime_error("action shard write failed");}fs::rename(temp,path);
}
inline ActionData load_actions(const fs::path& path) {
    std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);if(!f||(std::memcmp(magic,"BLZACT02",8)&&std::memcmp(magic,"BLZACT03",8)))throw std::invalid_argument("invalid action shard");ActionData a;a.architecture=magic[7]=='3'?placement_schema:action_schema;
    a.x=read_vector<float>(f,policy_inputs(a.architecture)*action_pool*65536ull);a.labels=read_vector<uint8_t>(f,action_pool*65536ull);a.offsets=read_vector<uint32_t>(f,65537);a.progress=read_vector<float>(f,65536);a.from=read_vector<uint32_t>(f,action_pool*65536ull);a.to=read_vector<uint32_t>(f,action_pool*65536ull);if(a.architecture==placement_schema)a.targets=read_vector<float>(f,9*action_pool*65536ull);
    if(f.peek()!=EOF)throw std::invalid_argument("unexpected action shard tail");validate_actions(a);return a;
}
inline EvalSettings action_eval(double pixels=32,double limit=3) {
    EvalSettings e;e.profile=Profile::Attributes;e.screen_size=pixels;e.limit=limit;e.max_changed_area=.5;
    e.views={6,2,0xB1172026};e.supersample=4;e.max_supersample=8;return e;
}
inline std::pair<Mesh,json> training_mesh(const std::string& id) {
    auto frozen=read_json("research/neural/training-manifest.json");bool allowed=false;
    for(auto& a:frozen.at("assets"))allowed|=a.at("id")==id;
    if(!allowed)throw std::invalid_argument("asset is outside frozen training selection");
    auto corpus=read_json("research/corpus.json");for(auto& a:corpus.at("assets"))if(a.at("id")==id){
        if(a.at("split")!="development")throw std::invalid_argument("training split mismatch");
        for(auto& f:a.at("files"))if(file_sha256(f.at("path").get<std::string>())!=f.at("sha256").get<std::string>())throw std::invalid_argument("training mesh checksum mismatch");
        auto m=load_mesh(a.at("path").get<std::string>());return {std::move(m),a};}
    throw std::invalid_argument("missing training asset");
}
}
