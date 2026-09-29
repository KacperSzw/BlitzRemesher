#pragma once
#include "blitz/neural.hpp"
#include <array>
#include <filesystem>
namespace blitz::neural {
constexpr uint32_t features=24, hidden=64, conditions=8, outputs=4, schema=1;
constexpr uint64_t max_raster_samples=64000000;
// Retain the original refinement sequence and initial sampling. Uncertain bounds
// at the last fitting resolution reject the candidate; no limit is relaxed.
uint8_t bounded_refinement(const EvalSettings&);
constexpr uint8_t Locked=1, Boundary=2, Seam=4, Material=8, Used=16;
struct Graph {
    std::vector<uint32_t> offsets,neighbors,canonical;
    std::vector<float> x; // vertex-major, 24 channels
    std::vector<uint8_t> flags;
    size_t size() const {return flags.size();}
};
Graph graph(MeshView);
struct Patch {
    Graph graph;
    std::vector<uint32_t> ids; // local -> source; first core entries own losses/output
    uint32_t core{};
};
Patch patch(const Graph&,std::span<const uint32_t> core,uint32_t maximum=262144);
std::array<float,conditions> condition(const EvalSettings&,double adjacent_limit,double fraction);
struct Prediction { std::vector<float> values; }; // keep logit and normalized xyz displacement
struct DecodeStats {uint64_t accepted{},rejected{};};
Lod decode(MeshView,const Graph&,const Prediction&,size_t target,OutputMode,DecodeStats* = nullptr,const std::function<bool()>& = {},std::vector<uint32_t>* representatives=nullptr);
// Shared portable float32 tensor layout: row-major [out,in], then bias.
struct WeightsData {
    std::vector<float> values;
    std::string provenance;
};
constexpr uint32_t layer_in[5]={features*2,hidden*2,hidden*2,hidden+conditions,hidden};
constexpr uint32_t layer_out[5]={hidden,hidden,hidden,hidden,outputs};
constexpr size_t weight_count=[] {size_t n=0;for(int i=0;i<5;++i)n+=size_t(layer_out[i])*(layer_in[i]+1);return n;}();
WeightsData load_weights(const std::filesystem::path&,std::string* hash=nullptr);
void save_weights(const std::filesystem::path&,const WeightsData&);
std::string sha256(std::span<const std::byte>);
std::string file_sha256(const std::filesystem::path&);
struct NumericLayer;
using NumericTrace=std::vector<NumericLayer>;
std::vector<float> encode_cuda(const Graph&,const WeightsData&,const NeuralOptions&,NumericTrace* = nullptr);
std::vector<float> encode_mesh_cuda(const Graph&,const WeightsData&,const NeuralOptions&,const std::function<bool()>& = {});
Prediction predict_cuda(std::span<const float>,const std::array<float,conditions>&,const WeightsData&,const NeuralOptions&,NumericTrace* = nullptr);
Raster raster_cuda(MeshView,const Bounds&,const Camera&,double,uint8_t,bool,const NeuralOptions& = {});
// Research-only endpoint labels; source representatives remain in source ID space.
Lod teacher(MeshView,const ReduceSettings&,std::vector<uint32_t>& representatives);
}
