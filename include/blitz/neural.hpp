#pragma once
#include "remesher.hpp"
#include <memory>
#include <stdexcept>
namespace blitz {
class NeuralUnavailable : public std::runtime_error { using std::runtime_error::runtime_error; };
struct NeuralOptions {
    int32_t device{};
    uint32_t memory_mib{6144}; // Includes library-owned CUDA scratch; minimum 128.
    bool overdraw_tiebreak{true};
    uint32_t action_trials{64}; // Per v2 proposal; additional measured visual trials.
};
enum class NeuralResourceLimit:uint8_t { None,SampleCount,WorkspaceMemory,DeviceMemory,TileEntries };
struct NeuralAuditFailure {
    double screen_pixels{};
    uint64_t requested{},limit{}; // Samples, bytes or tile entries according to kind.
    uint32_t view{};
    uint8_t supersample{};
    NeuralResourceLimit kind{};
};
struct NeuralStats {
    uint64_t encode_ns{},inference_ns{},decode_ns{},gpu_audit_ns{},reference_audit_ns{};
    uint64_t decoded{},legal_collapses{},rejected_collapses{},reference_rejections{};
    uint32_t fallback_levels{};
    uint32_t bounded_audits{},resource_failures{};
    uint64_t gpu_peak_bytes{};
    NeuralAuditFailure first_resource_failure{};
    uint64_t action_ranked{},action_trials{};
};
// Immutable weights. Calls use private CUDA workspaces and preserve the caller's device.
// Source streams must outlive the returned Result, as for generate().
class NeuralModel {
public:
    explicit NeuralModel(const char* file,const NeuralOptions& = {});
    ~NeuralModel();
    NeuralModel(NeuralModel&&) noexcept;
    NeuralModel& operator=(NeuralModel&&) noexcept;
    NeuralModel(const NeuralModel&)=delete;
    NeuralModel& operator=(const NeuralModel&)=delete;
    const std::string& sha256() const;
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
    friend Result generate_neural(MeshView,const Settings&,const NeuralModel&,NeuralStats*);
};
bool neural_available(int32_t device=0) noexcept;
Result generate_neural(MeshView,const Settings&,const NeuralModel&,NeuralStats* = nullptr);
// Same sampled metric/refinement as evaluate(); final CPU confirmation is used in generation.
Measurement evaluate_cuda(MeshView,MeshView,const Bounds&,const EvalSettings&,const NeuralOptions& = {},NeuralStats* = nullptr);
// Pre-depth geometric overlap using fixed centers and top-left fill, averaged over views.
double overlap_cuda(MeshView,const Bounds&,double,ViewSet,const NeuralOptions& = {});
}
