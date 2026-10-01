#pragma once
#include "blitz/neural.hpp"
#include "neural/device_view.hpp"
#include "neural/vertex_storage.hpp"
#include <array>
#include <filesystem>
namespace blitz::neural {
constexpr uint32_t features = 24, hidden = 64, conditions = 8, outputs = 4, schema = 1;
constexpr uint64_t max_raster_samples = 64000000;
// Retain the original refinement sequence and initial sampling. Uncertain bounds
// at the last fitting resolution reject the candidate; no limit is relaxed.
uint8_t bounded_refinement(const EvalSettings&);
constexpr uint8_t Locked = 1, Boundary = 2, Seam = 4, Material = 8, Used = 16;
struct Graph {
    std::vector<uint32_t> offsets, neighbors, canonical;
    std::vector<float> x; // vertex-major, 24 channels
    std::vector<uint8_t> flags;
    size_t size() const {
        return flags.size();
    }
};
Graph graph(MeshView);
struct Patch {
    Graph graph;
    std::vector<uint32_t> ids; // local -> source; first core entries own losses/output
    uint32_t core{};
};
Patch patch(const Graph&, std::span<const uint32_t> core, uint32_t maximum = 262144);
std::array<float, conditions> condition(const EvalSettings&, double adjacent_limit,
                                        double fraction);
struct Prediction {
    std::vector<float> values;
}; // keep logit and normalized xyz displacement
struct DecodeStats {
    uint64_t accepted{}, rejected{};
};
Lod decode(MeshView, const Graph&, const Prediction&, size_t target, OutputMode,
           DecodeStats* = nullptr, const std::function<bool()>& = {},
           std::vector<uint32_t>* representatives = nullptr);
// Shared portable float32 tensor layout: row-major [out,in], then bias.
struct WeightsData {
    std::vector<float> values;
    std::string provenance;
    uint32_t architecture{schema}, hidden_width{64};
};
constexpr uint32_t layer_in[5] = {features * 2, hidden * 2, hidden * 2, hidden + conditions,
                                  hidden};
constexpr uint32_t layer_out[5] = {hidden, hidden, hidden, hidden, outputs};
constexpr size_t weight_count = [] {
    size_t n = 0;
    for (int i = 0; i < 5; ++i)
        n += size_t(layer_out[i]) * (layer_in[i] + 1);
    return n;
}();
WeightsData load_weights(const std::filesystem::path&, std::string* hash = nullptr);
void save_weights(const std::filesystem::path&, const WeightsData&);
std::string sha256(std::span<const std::byte>);
std::string file_sha256(const std::filesystem::path&);
struct NumericLayer;
using NumericTrace = std::vector<NumericLayer>;
std::vector<float> encode_cuda(const Graph&, const WeightsData&, const NeuralOptions&,
                               NumericTrace* = nullptr);
std::vector<float> encode_mesh_cuda(const Graph&, const WeightsData&, const NeuralOptions&,
                                    const std::function<bool()>& = {});
Prediction predict_cuda(std::span<const float>, const std::array<float, conditions>&,
                        const WeightsData&, const NeuralOptions&, NumericTrace* = nullptr);
Raster raster_cuda(MeshView, const Bounds&, const Camera&, double, uint8_t, bool,
                   const NeuralOptions& = {});
Raster raster_gpu(MeshView, const Bounds&, const Camera&, double, uint8_t, bool,
                  const NeuralOptions&);
// Research-only quantization probe. Never participates in hard acceptance.
struct RasterPrecisionResult {
    Raster raster;
    double seconds{};
    uint8_t bytes_per_pixel{};
};
struct RasterBenchmarkResult {
    Raster raster;
    double seconds{}, setup_seconds{}, packing_seconds{}, render_seconds{}, unpack_seconds{};
    uint64_t draw_bytes{}, gpu_bytes{};
};
RasterBenchmarkResult raster_benchmark(MeshView, const Bounds&, const Camera&, double, uint8_t,
                                       bool, const NeuralOptions&, uint32_t repeats = 8,
                                       bool coverage_only = false);
struct DiagnosticRaster {
    Raster raster;
    std::vector<uint32_t> faces;
    std::vector<float> depth;
};
DiagnosticRaster diagnostic_raster(MeshView, const Bounds&, const Camera&, double, uint8_t, bool,
                                   const NeuralOptions&,
                                   const VertexBounds* quantization = nullptr);
struct DistanceFieldBenchmark {
    std::vector<float> squared;
    double seconds{};
    uint64_t bytes{};
};
DistanceFieldBenchmark benchmark_distance_field(std::span<const uint8_t> sites, uint32_t size,
                                                const NeuralOptions&, uint32_t repeats = 8);
RasterPrecisionResult raster_precision_cuda(MeshView, const Bounds&, const Camera&, double, uint8_t,
                                            bool, uint8_t attribute_bits, uint8_t depth_bits,
                                            uint8_t position_bits, const NeuralOptions& = {});
// Per-generation CUDA storage; fixed source streams are borrowed and immutable.
enum class AuditVerdict : uint8_t { Unknown, Pass, Fail };
// Bounds for teacher ordering, deliberately distinct from exact Measurement.
struct AuditPredicate {
    AuditVerdict verdict{AuditVerdict::Unknown};
    double error_upper{}, changed_area{};
    uint32_t views{};
    uint8_t supersample{};
    bool resource_limited{};
};
struct CandidateAudit {
    AuditPredicate value;
    uint32_t faces{};
    bool valid{}, pruned{};
};
// Test/research boundary: compare supplied images without invoking a rasterizer.
Measurement measure_rasters_cuda(const Raster&, const Raster&, const EvalSettings&,
                                 const NeuralOptions&);
AuditPredicate certify_rasters_cuda(const Raster&, const Raster&, const EvalSettings&,
                                    const NeuralOptions&, uint32_t queue_capacity = 262144);
class AuditCuda {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Measurement evaluate_device(MeshView, DeviceMeshView, const Bounds&, const EvalSettings&,
                                NeuralStats*, double, bool*, bool);

  public:
    explicit AuditCuda(const NeuralOptions&, MeshView fixed_source = {});
    ~AuditCuda();
    // Bind a validated immutable predecessor. Its streams must remain alive
    // and unchanged until the next binding or destruction of this evaluator.
    void bind_reference(MeshView);
    void clear_reference() noexcept;
    Measurement evaluate(MeshView, MeshView, const Bounds&, const EvalSettings&,
                         NeuralStats* = nullptr);
    // An optional teacher incumbent permits stopping once a completed view
    // proves this candidate cannot improve its normalized error/area margin.
    // A pruned result is a bound, never a pass/fail training label.
    Measurement evaluate(MeshView, DeviceMeshView, const Bounds&, const EvalSettings&,
                         NeuralStats* = nullptr,
                         double incumbent_margin = std::numeric_limits<double>::infinity(),
                         bool* pruned = nullptr);
    AuditPredicate certify(MeshView, DeviceMeshView, const Bounds&, const EvalSettings&,
                           NeuralStats* = nullptr,
                           double incumbent_area = std::numeric_limits<double>::infinity(),
                           bool* pruned = nullptr);
    std::vector<CandidateAudit>
    certify_candidates(MeshView, std::span<const DeviceMeshView>, const Bounds&,
                       const EvalSettings&, NeuralStats* = nullptr,
                       double incumbent_area = std::numeric_limits<double>::infinity());
};
// Explicit per-thread session: device/pipelines survive asset-local evaluators.
// The session must outlive every evaluator created in its scope.
class AuditSession {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit AuditSession(const NeuralOptions&);
    ~AuditSession();
    AuditSession(const AuditSession&) = delete;
};
// Research-only endpoint labels; source representatives remain in source ID space.
Lod teacher(MeshView, const ReduceSettings&, std::vector<uint32_t>& representatives);
} // namespace blitz::neural
