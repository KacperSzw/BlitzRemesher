#pragma once
#include "neural/placement.hpp"
#include "neural/vertex_storage.hpp"
namespace blitz::neural {
enum class TeacherSelection : uint8_t { GeometricRandom, PolicyMixed };
// Read-only diagnostic record. Actions borrow a host buffer only for the
// callback; failed backoff prefixes intentionally appear more than once.
struct ActionTrial {
    std::span<const Action> actions;
    uint32_t trial, iteration, faces_before, faces_after, first_rank, rank_cursor;
    bool geometry_valid, accepted;
};
using ActionTrialObserver = std::function<void(const ActionTrial&)>;
struct SupportedRanking {
    Action global, supported;
    float global_score{}, supported_score{};
    uint32_t seed{};
    uint8_t pool{};
    bool global_in_support{};
};
// Diagnostic only: the frozen collector constructs the same mixed action pool
// as teaching. It never supplies labels or a visual acceptance decision.
struct RankingSupport {
    ActionCuda& collector;
    std::function<void(const SupportedRanking&)> observe;
    uint32_t seed{101};
};
// CPU readback methods are reference-test/export boundaries. Mesh topology,
// features, legality and all intermediate action records live on the device.
class GpuActionState {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    // Episode seeds retain the source normalization and progress denominator.
    // The optional origin is borrowed only during construction.
    // inference_batch bounds feature scratch, never the number of ranked actions.
    GpuActionState(MeshView, const NeuralOptions&, bool free_placement = false,
                   const VertexBounds* quantization = nullptr, MeshView origin = {},
                   uint32_t inference_batch = 16384);
    ~GpuActionState();
    GpuActionState(const GpuActionState&) = delete;
    GpuActionState& operator=(const GpuActionState&) = delete;
    std::vector<ActionRecord> actions(const std::array<float, conditions>&);
    std::vector<PlacementRecord> placements(const std::array<float, conditions>&);
    std::vector<PlacementRecord>
    teacher_actions(const std::array<float, conditions>&, uint32_t count, uint32_t seed,
                    ActionCuda* policy = nullptr,
                    TeacherSelection = TeacherSelection::GeometricRandom,
                    std::vector<Placement>* decoded_policy = nullptr);
    struct Proposal {
        Placement placement;
        float target[9];
        uint8_t normal_mask;
    };
    std::vector<Proposal> teacher_proposals(Action, bool supervise_normals = true);
    // Independent placements share one compacted topology. Views expire at the
    // next trial attempt (even an invalid trial), trial_batch, or state mutation;
    // counts/validity remain on the device. Returned views never own storage.
    std::vector<DeviceMeshView> trial_batch(Action, std::span<const Proposal>);
    // Endpoint-only trial; the borrowed view has the same lifetime as trial_batch.
    bool trial(Action, DeviceMeshView&);
    bool trial(Action, const Placement&, DeviceMeshView&);
    void commit(Action, const Placement&);
    Lod snapshot();
    Lod trial(Action);
    void commit(Action);
    void reset();
    DeviceMeshView view() const;
    Lod execute(const std::array<float, conditions>&, size_t target, uint32_t budget, ActionCuda*,
                NeuralRanking, uint32_t seed, uint8_t batch,
                const std::function<bool(DeviceMeshView)>& gate, ActionStats*,
                const std::function<bool()>& cancelled = {},
                const std::function<void(GpuActionState&)>& observe = {},
                const ActionTrialObserver& trace = {}, const RankingSupport* support = nullptr);
};
} // namespace blitz::neural
