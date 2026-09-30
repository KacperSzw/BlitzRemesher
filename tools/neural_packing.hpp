#pragma once
#include "neural_action_data.hpp"
#include "neural_json.hpp"
#include <set>
#include "neural_quantization.hpp"
namespace blitz::neural::training {
inline json measurement_json(const Measurement& m){return {{"passed",m.passed},{"complete",m.complete},{"error",std::isfinite(m.error)?json(m.error):json(nullptr)},{"error_kind",std::isfinite(m.error)?"measured":"exceeds_limit_or_unavailable"},{"coverage",m.coverage},{"coverage_upper",m.coverage_upper},{"changed_area",m.changed_area},{"normal_degrees",m.normal_degrees},{"worst_view",m.worst_view},{"views",m.views_evaluated},{"sampling",m.supersample},{"resource_limited",m.resource_limited}};}
inline double diagnostic_cost(const Pixel& a,const Pixel& b,double spatial,const EvalSettings& e){
    if(!b.visible)return INFINITY;double cost=spatial;
    if(e.profile!=Profile::Coverage&&e.weights.normal>0){double angle=std::acos(std::clamp(dot(a.normal,b.normal)/std::max(1e-30,length(a.normal)*length(b.normal)),-1.,1.));cost+=std::pow(angle*e.weights.normal,2);}
    if(e.profile==Profile::Attributes){auto d=Vec3{a.color.x-b.color.x,a.color.y-b.color.y,a.color.z-b.color.z};cost+=e.weights.color*e.weights.color*dot(d,d);if(a.material!=b.material)cost+=e.weights.material*e.weights.material;}return cost;
}
struct PackingWitnesses {uint64_t unmatched{},ownership_changes{};json pixels=json::array();std::set<uint32_t> faces;};
// An independent CPU diagnostic over the actual GPU attachments. The bounded
// search reports failure witnesses, never pretends its cutoff is an exact error.
inline PackingWitnesses packing_witnesses(const DiagnosticRaster& a,const DiagnosticRaster& b,const EvalSettings& e){
    PackingWitnesses out;int size=int(a.raster.width),radius=int(std::ceil(e.limit*e.supersample));double inverse=1./(e.supersample*e.supersample),limit=e.limit*e.limit;
    for(size_t i=0;i<a.faces.size();++i)out.ownership_changes+=a.faces[i]!=b.faces[i];
    for(unsigned direction=0;direction<2;++direction){const auto& from=direction?b:a;const auto& to=direction?a:b;
        for(size_t i=0;i<from.raster.pixels.size();++i){const auto& p=from.raster.pixels[i];if(!p.visible)continue;double best=diagnostic_cost(p,to.raster.pixels[i],0,e);if(best<=limit)continue;int x=int(i%size),y=int(i/size);uint32_t match=UINT32_MAX;
            for(int yy=std::max(0,y-radius);yy<=std::min(size-1,y+radius);++yy)for(int xx=std::max(0,x-radius);xx<=std::min(size-1,x+radius);++xx){double spatial=((x-xx)*(x-xx)+(y-yy)*(y-yy))*inverse;if(spatial>limit||spatial>=best)continue;size_t j=size_t(yy)*size+xx;double cost=diagnostic_cost(p,to.raster.pixels[j],spatial,e);if(cost<best){best=cost;match=uint32_t(j);}}
            if(best<=limit)continue;++out.unmatched;if(from.faces[i]!=UINT32_MAX)out.faces.insert(from.faces[i]);if(to.faces[i]!=UINT32_MAX)out.faces.insert(to.faces[i]);
            if(out.pixels.size()<64){auto pixel=[&](const DiagnosticRaster& r,size_t j){const auto& q=r.raster.pixels[j];return json{{"face",r.faces[j]==UINT32_MAX?json(nullptr):json(r.faces[j])},{"depth",r.depth[j]},{"normal",{q.normal.x,q.normal.y,q.normal.z}},{"material",q.material},{"visible",q.visible},{"covered",q.covered}};};
                out.pixels.push_back({{"direction",direction?"candidate_to_source":"source_to_candidate"},{"xy",{x,y}},{"source",pixel(a,i)},{"candidate",pixel(b,i)},{"threshold",e.limit},{"nearest_within_spatial_window",std::isfinite(best)?json(std::sqrt(best)):json(nullptr)},{"match_pixel",match==UINT32_MAX?json(nullptr):json(match)}});}
        }
    }return out;
}
inline json diagnose_packing(MeshView source,MeshView candidate,const NeuralOptions& options,const EvalSettings& e){
    auto b=bounds(source);auto domain=vertex_bounds(source);auto views=cameras(b,e.screen_size,e.views);auto control=options;control.vertex_storage=NeuralVertexStorage::Float32;json rows=json::array();
    for(uint32_t view=0;view<views.size();++view){auto a=diagnostic_raster(source,b,views[view],e.screen_size,e.supersample,e.force_two_sided,control);auto q=diagnostic_raster(candidate,b,views[view],e.screen_size,e.supersample,e.force_two_sided,options,&domain);auto w=packing_witnesses(a,q,e);
        auto measured=measure_rasters_cuda(a.raster,q.raster,e,options);rows.push_back({{"view",view},{"audit",measurement_json(measured)},{"ownership_changes",w.ownership_changes},{"unmatched_pixels",w.unmatched},{"pixels",w.pixels},{"faces",w.faces}});
    }return {{"storage",storage_name(options.draw_storage())},{"pixels",e.screen_size},{"limit",e.limit},{"views",rows},{"score",nullptr}};
}
struct PackingRepair {Mesh mesh;Measurement initial,final;uint32_t trials{},changed_vertices{};json diagnostics;};
inline PackingRepair repair_packing(MeshView source,const NeuralOptions& options,const EvalSettings& e,uint32_t budget=64,MeshView prepared={}){
    auto repair=repair_packing_gpu(source,options,e,budget,prepared);json attempts=json::array();for(const auto& a:repair.attempts)attempts.push_back({{"trial",a.trial},{"view",a.view},{"vertex",a.vertex},{"axis",a.axis},{"direction",a.direction},{"unmatched_before",a.before},{"unmatched_after",a.after},{"kept",a.kept}});
    auto diagnostics=json{{"initial",measurement_json(repair.initial)},{"final",measurement_json(repair.final)},{"trials",repair.trials},{"changed_vertices",repair.changed_vertices},{"attempts",attempts}};
    return {std::move(repair.mesh),repair.initial,repair.final,repair.trials,repair.changed_vertices,std::move(diagnostics)};
}
}
