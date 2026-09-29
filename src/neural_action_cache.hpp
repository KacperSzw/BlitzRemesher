#pragma once
#include "neural_action.hpp"
#include <tuple>
namespace blitz::neural {
// Rejected endpoint candidates borrow the same immutable source streams. Cache
// exact index/material sequences only; a hash collision can never reject a mesh.
class ActionRejections {
public:
    explicit ActionRejections(size_t bytes=4u<<20):limit_(bytes) {}
    void configure(const EvalSettings& e) {
        Key key{e.screen_size,e.limit,e.max_changed_area,e.weights.normal,e.weights.color,e.weights.material,
            e.views.rotation_seed,e.views.orthographic,e.views.perspective,e.supersample,e.max_supersample,e.profile,e.force_two_sided,e.force_scalar};
        if(key!=key_){entries_.clear();bytes_=0;key_=key;}
    }
    bool contains(MeshView m) const {
        for(const auto& e:entries_)if(e.indices.size()==m.indices.size()&&e.materials.size()==m.materials.size()&&
            std::equal(e.indices.begin(),e.indices.end(),m.indices.begin())&&std::equal(e.materials.begin(),e.materials.end(),m.materials.begin()))return true;
        return false;
    }
    void insert(MeshView m) {
        auto bytes=m.indices.size_bytes()+m.materials.size_bytes();
        if(entries_.size()>=64||bytes>limit_-bytes_||contains(m))return;
        Entry e;e.indices.assign(m.indices.begin(),m.indices.end());e.materials.assign(m.materials.begin(),m.materials.end());
        bytes=e.indices.capacity()*sizeof(uint32_t)+e.materials.capacity()*sizeof(uint16_t);
        if(bytes>limit_-bytes_)return;bytes_+=bytes;entries_.push_back(std::move(e));
    }
private:
    using Key=std::tuple<double,double,double,double,double,double,uint32_t,uint16_t,uint16_t,uint8_t,uint8_t,Profile,bool,bool>;
    struct Entry {std::vector<uint32_t> indices;std::vector<uint16_t> materials;};
    Key key_{};size_t limit_,bytes_{};std::vector<Entry> entries_;
};
}
