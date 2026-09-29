#pragma once
#include "neural_internal.hpp"
#include <tuple>
namespace blitz::neural {
template<class T> bool same_storage(Stream<T> a,Stream<T> b) {
    return a.count==b.count&&(!a.count||(a.data==b.data&&a.stride==b.stride));
}
inline bool source_attributes(MeshView a,MeshView source) {
    return same_storage(a.positions,source.positions)&&same_storage(a.normals,source.normals)&&same_storage(a.colors,source.colors)
        &&same_storage(a.uv,source.uv)&&same_storage(a.tangents,source.tangents)
        &&std::equal(a.double_sided.begin(),a.double_sided.end(),source.double_sided.begin(),source.double_sided.end());
}
// Only endpoint meshes borrowing immutable source attributes qualify. Exact
// owned index/material keys avoid stale pointers and hash-dependent acceptance.
class AuditMemo {
    using Key=std::tuple<double,double,double,double,double,double,uint32_t,uint16_t,uint16_t,uint8_t,uint8_t,Profile,bool,bool,float,float,float,double>;
    static Key key(const Bounds& b,const EvalSettings& e) {return {e.screen_size,e.limit,e.max_changed_area,e.weights.normal,e.weights.color,e.weights.material,
        e.views.rotation_seed,e.views.orthographic,e.views.perspective,e.supersample,e.max_supersample,e.profile,e.force_two_sided,e.force_scalar,b.center.x,b.center.y,b.center.z,b.radius};}
    struct Topology {
        std::vector<uint32_t> indices;std::vector<uint16_t> materials;
        bool equals(MeshView m) const {return std::equal(indices.begin(),indices.end(),m.indices.begin(),m.indices.end())&&std::equal(materials.begin(),materials.end(),m.materials.begin(),m.materials.end());}
        void assign(MeshView m){indices.assign(m.indices.begin(),m.indices.end());materials.assign(m.materials.begin(),m.materials.end());}
        size_t bytes() const {return indices.capacity()*sizeof(uint32_t)+materials.capacity()*sizeof(uint16_t);}
    };
    struct Entry {Key key;Topology a,b;Measurement result;};
    MeshView source_;size_t limit_,used_{};std::vector<Entry> entries_;
public:
    explicit AuditMemo(MeshView source,size_t limit=4u<<20):source_(source),limit_(limit) {}
    const Measurement* find(MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& e) const {
        if(!source_attributes(a,source_)||!source_attributes(b,source_))return nullptr;
        auto k=key(bounds,e);for(const auto& row:entries_)if(row.key==k&&row.a.equals(a)&&row.b.equals(b))return &row.result;return nullptr;
    }
    void insert(MeshView a,MeshView b,const Bounds& bounds,const EvalSettings& e,const Measurement& result) {
        if(!result.complete||!result.passed||result.resource_limited||!std::isfinite(result.error)||!std::isfinite(result.changed_area)
          ||!source_attributes(a,source_)||!source_attributes(b,source_)||entries_.size()>=64||find(a,b,bounds,e))return;
        auto payload=a.indices.size_bytes()+b.indices.size_bytes()+a.materials.size_bytes()+b.materials.size_bytes();
        const auto capacity=entries_.size()==entries_.capacity()?std::max(size_t(4),entries_.capacity()*2):entries_.capacity();
        const auto growth=capacity==entries_.capacity()?0:capacity*sizeof(Entry);
        if(payload+growth>limit_-used_)return;
        Entry row{key(bounds,e),{},{},result};row.a.assign(a);row.b.assign(b);payload=row.a.bytes()+row.b.bytes();
        if(payload+growth>limit_-used_)return;
        if(growth){auto old=entries_.capacity();entries_.reserve(capacity);used_+=(entries_.capacity()-old)*sizeof(Entry);}
        used_+=payload;entries_.push_back(std::move(row));
    }
};
}
