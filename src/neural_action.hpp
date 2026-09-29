#pragma once
#include "neural_internal.hpp"

namespace blitz::neural {
constexpr uint32_t action_schema=2, action_features=80, action_outputs=3;
constexpr uint32_t action_layer_in[3]={action_features,hidden,hidden};
constexpr uint32_t action_layer_out[3]={hidden,hidden,action_outputs};
constexpr size_t action_weight_count=hidden*(action_features+1)+hidden*(hidden+1)+action_outputs*(hidden+1);
struct Action {uint32_t from{},to{},revision{};};
struct ActionRecord {Action action;std::array<float,action_features> x{};};

// Borrows immutable input streams; owns indices and topology. Each accepted edit
// invalidates actions from the preceding revision. Trials cannot modify the state.
class ActionState {
public:
    explicit ActionState(MeshView);
    MeshView view() const {return current_.view(input_);}
    const Lod& lod() const {return current_;}
    std::vector<ActionRecord> actions(const std::array<float,conditions>&) const;
    bool legal(Action) const;
    Lod trial(Action) const;
    void commit(Action);
    double teacher_cost(Action) const; // Offline control only; never called by inference.
private:
    struct Edge {uint32_t a,b;uint8_t count;};
    MeshView input_;
    Lod current_;
    Graph graph_;
    uint32_t revision_{};
    double diameter_{};
    std::vector<uint32_t> geometry_,offsets_,neighbors_,face_offsets_,faces_;
    std::vector<Edge> edges_;
    std::vector<uint8_t> boundary_,invalid_;
    void rebuild();
    bool mapping(Action,std::vector<std::pair<uint32_t,uint32_t>>&) const;
};
struct ActionStats {uint64_t ranked{},trials{},accepted{},rejected{};};
using ActionRanker=std::function<std::vector<float>(const ActionState&,std::span<const ActionRecord>)>;
using ActionGate=std::function<bool(MeshView)>;
// One per inference call/thread. Weights and bounded scratch are reused across
// changing mesh states; inputs/output are row-major. No LibTorch at runtime.
class ActionCuda {
public:
    ActionCuda(const WeightsData&,const NeuralOptions&,uint32_t batch=16384);
    ~ActionCuda();
    std::vector<float> predict(std::span<const float>);
    ActionCuda(const ActionCuda&)=delete;
    ActionCuda& operator=(const ActionCuda&)=delete;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
Lod execute_actions(MeshView,const std::array<float,conditions>&,size_t target,
    uint32_t trial_budget,const ActionRanker&,const ActionGate&,ActionStats* = nullptr,
    const std::function<bool()>& cancelled={});
}
