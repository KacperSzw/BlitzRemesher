#pragma once
#include "neural_placement.hpp"
namespace blitz::neural {
// CPU readback methods are reference-test/export boundaries. Mesh topology,
// features, legality and all intermediate action records live on the device.
class GpuActionState {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    GpuActionState(MeshView,const NeuralOptions&,bool free_placement=false);
    ~GpuActionState();
    GpuActionState(const GpuActionState&)=delete;
    GpuActionState& operator=(const GpuActionState&)=delete;
    std::vector<ActionRecord> actions(const std::array<float,conditions>&);
    std::vector<PlacementRecord> placements(const std::array<float,conditions>&);
    std::vector<PlacementRecord> teacher_actions(const std::array<float,conditions>&,uint32_t count,uint32_t seed,ActionCuda* policy=nullptr);
    struct Proposal {Placement placement;float target[9];uint8_t normal_mask;};
    std::vector<Proposal> teacher_proposals(Action);
    bool trial(Action,const Placement&,DeviceMeshView&);
    void commit(Action,const Placement&);
    Lod snapshot();
    Lod trial(Action);
    void commit(Action);
    void reset();
    DeviceMeshView view()const;
    Lod execute(const std::array<float,conditions>&,size_t target,uint32_t budget,
        ActionCuda*,NeuralRanking,uint32_t seed,uint8_t batch,
        const std::function<bool(DeviceMeshView)>& gate,ActionStats*,const std::function<bool()>& cancelled={});
};
}
