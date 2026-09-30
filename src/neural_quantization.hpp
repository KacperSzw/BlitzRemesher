#pragma once
#include "neural_internal.hpp"
namespace blitz::neural {
struct PackingAttempt {uint32_t trial,view,vertex,axis;int direction;uint64_t before,after;bool kept;uint32_t patch_face{UINT32_MAX};};
struct PackingRepairResult {Mesh mesh;Measurement initial,final;uint32_t trials{},changed_vertices{},failed_check{};std::vector<PackingAttempt> attempts;};
struct PackingAudit {MeshView reference;EvalSettings settings;};
inline bool same_packing_settings(const EvalSettings& a,const EvalSettings& b){return a.screen_size==b.screen_size&&a.limit==b.limit&&a.max_changed_area==b.max_changed_area&&a.profile==b.profile&&a.weights.normal==b.weights.normal&&a.weights.color==b.weights.color&&a.weights.material==b.weights.material&&a.views.orthographic==b.views.orthographic&&a.views.perspective==b.views.perspective&&a.views.rotation_seed==b.views.rotation_seed&&a.supersample==b.supersample&&a.max_supersample==b.max_supersample&&a.force_scalar==b.force_scalar&&a.force_two_sided==b.force_two_sided;}
PackingRepairResult repair_packing_gpu(MeshView,const NeuralOptions&,std::span<const PackingAudit>,uint32_t budget=256,MeshView prepared={});
// Cold preparation only. Every kept grid edit receives the unchanged complete
// audit against the immutable source. This bounded search is not a feasibility
// proof; failure remains visible to the caller.
PackingRepairResult repair_packing_gpu(MeshView,const NeuralOptions&,const EvalSettings&,uint32_t budget=64,MeshView prepared={},const EvalSettings* additional=nullptr);
}
