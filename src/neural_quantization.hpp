#pragma once
#include "neural_internal.hpp"
namespace blitz::neural {
struct PackingAttempt {uint32_t trial,view,vertex,axis;int direction;uint64_t before,after;bool kept;};
struct PackingRepairResult {Mesh mesh;Measurement initial,final;uint32_t trials{},changed_vertices{};std::vector<PackingAttempt> attempts;};
// Cold preparation only. Every kept grid edit receives the unchanged complete
// audit against the immutable source. This bounded search is not a feasibility
// proof; failure remains visible to the caller.
PackingRepairResult repair_packing_gpu(MeshView,const NeuralOptions&,const EvalSettings&,uint32_t budget=64,MeshView prepared={},const EvalSettings* additional=nullptr);
}
