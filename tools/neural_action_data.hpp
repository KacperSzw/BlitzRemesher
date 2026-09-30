#pragma once
#include "neural_action_types.hpp"
#include "neural_compact_data.hpp"
namespace blitz::neural::training {
inline void save_actions(const fs::path& path,const ActionData& a,bool compact=true) {
    static_assert(std::endian::native==std::endian::little);validate_actions(a);auto temp=path;temp+=".part";
    {std::ofstream f(temp,std::ios::binary);if(compact)write_compact(f,compact_actions(a));else{f.write(a.architecture==action_schema?"BLZACT02":"BLZACT03",8);write_vector(f,a.x);write_vector(f,a.labels);write_vector(f,a.offsets);write_vector(f,a.progress);write_vector(f,a.from);write_vector(f,a.to);if(a.architecture==placement_schema)write_vector(f,a.targets);}f.close();if(!f)throw std::runtime_error("action shard write failed");}fs::rename(temp,path);
}
inline ActionData load_actions(const fs::path& path) {
    std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);if(!f)throw std::invalid_argument("invalid action shard");if(!std::memcmp(magic,"BLZACT04",8))return expand_compact(read_compact(f));if(std::memcmp(magic,"BLZACT02",8)&&std::memcmp(magic,"BLZACT03",8))throw std::invalid_argument("invalid action shard");ActionData a;a.architecture=magic[7]=='3'?placement_schema:action_schema;
    a.x=read_vector<float>(f,policy_inputs(a.architecture)*action_pool*65536ull);a.labels=read_vector<uint8_t>(f,action_pool*65536ull);a.offsets=read_vector<uint32_t>(f,65537);a.progress=read_vector<float>(f,65536);a.from=read_vector<uint32_t>(f,action_pool*65536ull);a.to=read_vector<uint32_t>(f,action_pool*65536ull);if(a.architecture==placement_schema)a.targets=read_vector<float>(f,9*action_pool*65536ull);
    if(f.peek()!=EOF)throw std::invalid_argument("unexpected action shard tail");validate_actions(a);return a;
}
inline CompactActions load_compact_actions(const fs::path& path){
    std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);if(f&&!std::memcmp(magic,"BLZACT04",8))return read_compact(f);return compact_actions(load_actions(path));
}
inline EvalSettings action_eval(double pixels=32,double limit=3) {
    EvalSettings e;e.profile=Profile::Attributes;e.screen_size=pixels;e.limit=limit;e.max_changed_area=.5;
    e.views={6,2,0xB1172026};e.supersample=4;e.max_supersample=8;return e;
}
inline json training_metadata(const std::string& id,const fs::path& source="research/corpus.json",const fs::path& selection="research/neural/training-manifest.json") {
    auto frozen=read_json(selection);bool allowed=false;
    for(auto& a:frozen.at("assets"))allowed|=a.at("id")==id;
    if(!allowed)throw std::invalid_argument("asset is outside frozen training selection");
    auto corpus=read_json(source);for(auto& a:corpus.at("assets"))if(a.at("id")==id){
        if(a.at("split")!="development")throw std::invalid_argument("training split mismatch");
        for(auto& f:a.at("files"))if(file_sha256(f.at("path").get<std::string>())!=f.at("sha256").get<std::string>())throw std::invalid_argument("training mesh checksum mismatch");
        return a;}
    throw std::invalid_argument("missing training asset");
}
inline std::pair<Mesh,json> training_mesh(const std::string& id) {auto metadata=training_metadata(id);return {load_mesh(metadata.at("path").get<std::string>()),metadata};}
}
