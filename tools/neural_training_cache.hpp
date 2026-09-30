#pragma once
#include "neural_mesh_cache.hpp"
#include <map>
#include <mutex>
namespace blitz::neural::training {
// Shared immutable host meshes. Hash source files once at startup; cache misses
// verify the prepared payload before borrowing it. In-use entries cannot evict.
class TrainingCache {
    struct Entry {json metadata;std::shared_ptr<const PreparedMesh> mesh;size_t bytes{};uint64_t used{};};
    std::map<std::string,Entry> entries_;std::map<std::string,std::string> verified_;fs::path cache_;size_t limit_,live_{};uint64_t clock_{},hits_{},loads_{};std::mutex mutex_;
    static size_t bytes(const Mesh& m){auto sum=[](const auto& v){return v.capacity()*sizeof(v[0]);};return sum(m.positions)+sum(m.normals)+sum(m.uv)+sum(m.tangents)+sum(m.colors)+sum(m.indices)+sum(m.materials)+sum(m.double_sided)+sum(m.exact_position_bits);}
public:
    std::string corpus_hash,selection_hash,protocol_hash,binary_hash;
    TrainingCache(const fs::path& corpus,const fs::path& selection,const fs::path& cache,size_t limit):cache_(cache),limit_(limit){
        corpus_hash=file_sha256(corpus);selection_hash=file_sha256(selection);protocol_hash=file_sha256("research/PROTOCOL.md");binary_hash=file_sha256("/proc/self/exe");
        auto allowed=read_json(selection),all=read_json(corpus);
        for(const auto& row:allowed.at("assets")){auto id=row.at("id").get<std::string>();auto it=std::find_if(all.at("assets").begin(),all.at("assets").end(),[&](const auto& a){return a.at("id")==id;});
            if(it==all.at("assets").end()||it->at("split")!="development")throw std::invalid_argument("training cache split/selection mismatch");
            if(!entries_.emplace(id,Entry{*it}).second)throw std::invalid_argument("duplicate training cache asset");}
    }
    const json& metadata(const std::string& id)const{return entries_.at(id).metadata;}
    std::shared_ptr<const PreparedMesh> get(const std::string& id){
        std::lock_guard lock(mutex_);auto& entry=entries_.at(id);entry.used=++clock_;if(entry.mesh){++hits_;return entry.mesh;}
        // The cycle preloads its complete curriculum before starting workers.
        // Hash those inputs once, without reading unused selection members.
        for(const auto& f:entry.metadata.at("files")){auto path=f.at("path").get<std::string>(),hash=f.at("sha256").get<std::string>();auto v=verified_.find(path);
            if(v!=verified_.end()){if(v->second!=hash)throw std::invalid_argument("conflicting training source checksum");}
            else {if(file_sha256(path)!=hash)throw std::invalid_argument("training cache source checksum mismatch");verified_.emplace(path,hash);}}
        auto mesh=std::make_shared<PreparedMesh>(prepared_mesh(entry.metadata,cache_));auto required=bytes(mesh->source)+bytes(mesh->draw);
        if(required>limit_)throw std::length_error("one prepared mesh exceeds host cache budget");
        while(required>limit_-live_){Entry* oldest=nullptr;for(auto& [name,e]:entries_)if(e.mesh&&e.mesh.use_count()==1&&(!oldest||e.used<oldest->used))oldest=&e;
            if(!oldest)throw std::length_error("in-flight meshes exceed host cache budget");live_-=oldest->bytes;oldest->mesh.reset();}
        live_+=required;entry.bytes=required;entry.mesh=std::move(mesh);++loads_;return entry.mesh;
    }
    json stats(){std::lock_guard lock(mutex_);return {{"bytes",live_},{"limit",limit_},{"hits",hits_},{"loads",loads_},{"assets",entries_.size()}};}
};
}
