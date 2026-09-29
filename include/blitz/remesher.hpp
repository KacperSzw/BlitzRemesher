#pragma once
#include "evaluate.hpp"
#include <memory>
#include <optional>
namespace blitz {
enum class OutputMode:uint8_t { Rebuild,Reuse };
enum class ChainMode:uint8_t { Direct,Progressive,Hybrid };
enum class Objective:uint8_t { Quadric,Regularized,Visual,TopologyRelaxed };
enum class Status:uint8_t { Complete,BudgetLimited,Cancelled };
enum class ChainObjective:uint8_t { WholeChain,TailFirst };
enum class AppearanceStage:uint8_t { Off,Ordering,Attributes,Position };
struct Curve { std::vector<Vec2> points{{0,2},{1,3}}; double at(double) const; };
// Forced whole-chain modes are research controls, not production policies.
struct ResearchOptions {
    std::optional<OutputMode> output;
    ChainMode chain{ChainMode::Hybrid};
    double boundary_weight{};
    uint16_t coverage_cache_mib{256}; // 0 disables; 0..256 MiB, split equally between references and candidate.
    uint8_t graph_passes{}; // 0: incumbent search; 1..3: bounded whole-chain improvement passes.
    AppearanceStage appearance_stage{};
    bool conservative_screen{};
    bool density_targets{}; // Estimate rebuilt targets from referenced/output vertex density.
    bool merge_wedges{}; // Fit/merge continuous interior wedges after positional contractions.
    bool shared_rebuild{}; // Share exact source tuples; charge only changed rebuilt vertices.
    bool boundary_placement{},adaptive_targets{},component_candidates{},trace{},independent_seams{},topology_fallback{};
};
struct Settings {
    uint8_t levels{8}; Objective objective{Objective::Quadric};
    uint16_t triangle_overhead_bps{}; // 100 basis points = 1%; range 0..10000.
    std::optional<uint32_t> max_added_vertex_bytes_bps{2000}; // Relative to packed source vertex bytes; null disables.
    double pixels_per_meter{512},meters_per_unit{1};
    std::optional<double> base_pixels{},max_lod0_delta_px{};
    double last_pixels{16}; Curve transition{};
    Curve normal_importance{{{0,1},{1,1}}},attribute_importance{{{0,1},{1,1}}};
    Profile profile{Profile::Normals}; Weights weights{};
    ViewSet search_views{42,12,0xB1172024},audit_views{};
    uint8_t search_supersample{4},audit_supersample{8},max_supersample{32};
    double max_changed_area{1.0}; // Maximum 1 - mask IoU on both source and adjacent full audits.
    uint16_t candidate_budget{64}; uint8_t beam_width{8};
    bool prune{true},force_scalar{},coupled_wedges{true}; std::function<bool()> cancelled;
    PerformanceStats* performance{}; // Borrowed; reset at the start of generate().
    ResearchOptions research{};
};
struct ScheduleEntry { double pixels{},transition{},source{}; };
std::vector<ScheduleEntry> schedule(const Bounds&,const Settings&);
std::string validate(const Settings&);
struct Lod {
    Mesh data; ScheduleEntry schedule{}; Measurement adjacent{},source_error{};
    std::shared_ptr<const Mesh> vertex_pool; // One immutable buffer owner, never one object per vertex.
    uint32_t source_prefix_vertices{};
    // reduce(): relative to that call's input; generate(): relative to Result.source.
    bool shared_vertices{true};
    MeshView view(MeshView source) const {
        auto v=shared_vertices?source:vertex_pool?vertex_pool->view():data.view();
        v.indices=data.indices;v.materials=data.materials;return v;
    }
};
struct ProposalTrace {
    uint32_t input_triangles{},parent_triangles{},requested{},achieved{};
    uint64_t attempts{},collapsed{},geometry_rejections{},uv_rejections{},link_rejections{};
    double seconds{};
    uint8_t level{},origin{},strategy{},gate{},pass{}; // pass: baseline=0, adaptive retry=1, graph=2..4; origin: direct=0, progressive=1, tail probe=2.
    // gate: accepted=0, four gates=1..4, invalid=5, growth=6, duplicate=7, unavailable=8, vertex budget=9, objective bound=10.
};
struct StorageStats {
    uint64_t source_vertex_bytes{},added_vertex_bytes{},index_bytes{};
    uint64_t total() const { return source_vertex_bytes+added_vertex_bytes+index_bytes; }
};
struct ChainCost {
    std::vector<uint32_t> triangles;
    StorageStats storage;
};
struct ChainSelection { size_t reference{},selected{}; };
// Deterministic selection from an already audited pool. Does not establish validity.
ChainSelection select_chain(std::span<const ChainCost>,uint16_t overhead_bps,std::optional<uint64_t> added_vertex_budget_bytes={},ChainObjective=ChainObjective::WholeChain);
struct SearchProgress {
    uint64_t candidate_evaluations{},audit_evaluations{},triangle_total{},added_vertex_bytes{};
    double seconds{};
    uint8_t pass{}; // 0 is the complete incumbent; later entries are complete graph passes.
};
// Owned description of the finite visual contract, also exported in lods.json.
struct AuditContract {
    Profile profile{Profile::Normals};
    ViewSet search_views{},audit_views{};
    Weights weights{};
    Curve normal_importance{},attribute_importance{};
    uint8_t search_supersample{},audit_supersample{},max_supersample{};
};
uint64_t vertex_bytes(MeshView); // Canonical packed attributes, excluding borrowed stride padding.
uint64_t added_vertex_bytes(const Lod&,MeshView source);
struct Result {
    MeshView source; Bounds reference_bounds; std::vector<Lod> lods;
    AuditContract audit;
    Status status{Status::Complete}; uint64_t candidate_evaluations{};
    // Source search, adjacent search, source audit, adjacent audit.
    std::array<uint64_t,4> rejected_gates{};
    std::array<uint64_t,4> audit_evaluations{}; // Includes search screens and full audits, in the same order.
    std::array<uint64_t,4> area_rejected_gates{}; // Audit-only subset: pixel limit passed, area limit failed.
    std::array<Measurement,4> worst_rejected{};
    std::vector<ProposalTrace> proposals;
    uint64_t duplicate_proposals{},component_builds{},component_unavailable{},topology_fallback_proposals{},vertex_budget_rejections{};
    uint8_t tail_probe_evaluations{}; // At most eight direct tail probes per pass.
    uint64_t tail_reserved_vertex_bytes{};
    uint64_t transition_reconnections{}; // Extra audited edges, not reduction proposals.
    uint64_t adaptive_retry_evaluations{};
    bool adaptive_retry_attempted{},adaptive_retry_selected{};
    ChainObjective chain_objective{ChainObjective::TailFirst};
    uint64_t graph_candidates{},graph_edges{},graph_pruned_candidates{},graph_pruned_paths{};
    uint8_t graph_passes_completed{};
    std::vector<SearchProgress> search_progress;
    uint16_t triangle_overhead_bps{};
    std::optional<uint32_t> max_added_vertex_bytes_bps;
    std::optional<uint64_t> added_vertex_budget_bytes;
    double max_changed_area{1.0}; // Audit contract used for this result.
    ChainSelection selection;
    std::vector<ChainCost> candidates; // Final audited pool; no duplicate geometry payloads.
};
StorageStats storage_stats(const Result&); // Same buffer layout and duplicate policy as save_chain().
// First scheduled slot of each consecutive group with identical render data.
// Scheduled slots and their independent audit records remain unchanged (at most 32).
std::vector<uint8_t> runtime_levels(const Result&);
struct RuntimeLevelStorage {
    uint8_t scheduled_index{};
    uint64_t added_vertex_bytes{},index_bytes{},cumulative_added_vertex_bytes{};
};
std::vector<RuntimeLevelStorage> runtime_storage(const Result&);
struct ReductionStats {
    uint64_t appearance_bytes{}; // Coefficients, wedge lists and active attribute storage.
    uint64_t attempts{},collapsed{},geometry_rejections{},uv_rejections{},link_rejections{};
    uint64_t solve_attempts{},singular_solves{},nonfinite_solves{},position_fallbacks{},nonfinite_costs{};
    uint32_t initial_triangles{},final_triangles{},last_candidates{},last_locked_edges{},first_locked_edges{};
    uint8_t passes{}; // The reducer has at most 128 collapse passes.
};
struct ReductionStorage { uint8_t quadric_bytes{},candidate_bytes{}; };
ReductionStorage reduction_storage(); // Reports record sizes for diagnostics.
struct ReduceSettings {
    OutputMode output{OutputMode::Rebuild}; Objective objective{Objective::Quadric};
    size_t target_triangles{}; double normal_weight{1},regularization{1e-5};
    bool prune{},coupled_wedges{}; std::function<bool()> cancelled;
    ReductionStats* statistics{}; // Optional borrowed diagnostic output, reset per reduction.
    double boundary_weight{};
    bool boundary_placement{};
    bool independent_seams{}; // Collapse within each original chart; never weld attribute vertices.
    bool merge_wedges{};
    bool preserve_positions{}; // Keep untouched/endpoint coordinates byte exact through normalization.
    AppearanceStage appearance_stage{};
    Weights appearance_weights{};
    double screen_size{1}; // Pixels per source bounding-sphere diameter.
};
Lod reduce(MeshView,const ReduceSettings&);
using Proposer=std::function<Lod(MeshView,const ReduceSettings&)>;
// Optional research provider: borrowed output is relative to the provided input,
// which can be a rebuilt predecessor. generate() resolves its ownership and
// sends every changed candidate through the independent visual gates.
Result generate(MeshView,const Settings&,const Proposer& = {});
}
