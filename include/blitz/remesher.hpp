#pragma once
#include "evaluate.hpp"
#include <optional>
namespace blitz {
enum class OutputMode:uint8_t { Rebuild,Reuse };
enum class ChainMode:uint8_t { Direct,Progressive,Hybrid };
enum class Objective:uint8_t { Quadric,Regularized,Visual };
enum class Status:uint8_t { Complete,BudgetLimited,Cancelled };
struct Curve { std::vector<Vec2> points{{0,2},{1,3}}; double at(double) const; };
struct Settings {
    uint8_t levels{8}; OutputMode output{OutputMode::Rebuild};
    ChainMode chain{ChainMode::Hybrid}; Objective objective{Objective::Quadric};
    double pixels_per_meter{512},meters_per_unit{1};
    std::optional<double> base_pixels{},max_lod0_delta_px{};
    double last_pixels{16}; Curve transition{};
    Curve normal_importance{{{0,1},{1,1}}},attribute_importance{{{0,1},{1,1}}};
    Profile profile{Profile::Normals}; Weights weights{};
    ViewSet search_views{42,12,0xB1172024},audit_views{};
    uint8_t search_supersample{4},audit_supersample{8},max_supersample{32};
    uint16_t candidate_budget{64}; uint8_t beam_width{8};
    bool prune{true},force_scalar{},coupled_wedges{true}; std::function<bool()> cancelled;
};
struct ScheduleEntry { double pixels{},transition{},source{}; };
std::vector<ScheduleEntry> schedule(const Bounds&,const Settings&);
std::string validate(const Settings&);
struct Lod {
    Mesh data; ScheduleEntry schedule{}; Measurement adjacent{},source_error{};
    bool shared_vertices{true};
    MeshView view(MeshView source) const {
        if(!shared_vertices) return data.view();
        source.indices=data.indices; source.materials=data.materials; return source;
    }
};
struct Result {
    MeshView source; Bounds reference_bounds; std::vector<Lod> lods;
    Status status{Status::Complete}; uint64_t candidate_evaluations{};
    // Source search, adjacent search, source audit, adjacent audit.
    std::array<uint64_t,4> rejected_gates{};
    std::array<Measurement,4> worst_rejected{};
};
struct ReduceSettings {
    OutputMode output{OutputMode::Rebuild}; Objective objective{Objective::Quadric};
    size_t target_triangles{}; double normal_weight{1},regularization{1e-5};
    bool prune{},coupled_wedges{}; std::function<bool()> cancelled;
};
Lod reduce(MeshView,const ReduceSettings&);
using Proposer=std::function<Lod(MeshView,const ReduceSettings&)>;
// Optional research proposal provider. All candidates still pass the independent visual gates.
Result generate(MeshView,const Settings&,const Proposer& = {});
}
