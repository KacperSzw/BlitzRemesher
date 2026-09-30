#pragma once
#include "neural_json.hpp"
#include <filesystem>
#include <fstream>
#include <openssl/sha.h>
inline nlohmann::json audit_measurement(const blitz::Measurement& m){return {{"passed",m.passed},{"complete",m.complete},{"resource_limited",m.resource_limited},{"error",m.error},{"coverage",m.coverage},{"coverage_upper",m.coverage_upper},{"changed_area",m.changed_area},{"normal_degrees",m.normal_degrees},{"views",m.views_evaluated},{"worst_view",m.worst_view},{"area_worst_view",m.changed_area_worst_view},{"supersample",m.supersample}};}
inline nlohmann::json audit_mesh(const blitz::Mesh& m){
    nlohmann::json j={{"indices",m.indices},{"materials",m.materials},{"double_sided",m.double_sided}};
    auto fill=[&](const char* key,const auto& values,auto pack){j[key]=nlohmann::json::array();for(const auto& v:values)j[key].push_back(pack(v));};
    auto xyz=[](auto v){return nlohmann::json::array({v.x,v.y,v.z});};
    fill("positions",m.positions,xyz);fill("normals",m.normals,xyz);
    fill("uv",m.uv,[](auto v){return nlohmann::json::array({v.x,v.y});});
    fill("tangents",m.tangents,[](auto v){return nlohmann::json::array({v.x,v.y,v.z,v.w});});
    fill("colors",m.colors,[](auto v){return nlohmann::json::array({v.r,v.g,v.b,v.a});});return j;
}
inline blitz::Mesh audit_mesh(const nlohmann::json& j){
    blitz::Mesh m;for(auto& v:j.at("positions"))m.positions.push_back({v.at(0),v.at(1),v.at(2)});
    for(auto& v:j.at("normals"))m.normals.push_back({v.at(0),v.at(1),v.at(2)});
    for(auto& v:j.at("uv"))m.uv.push_back({v.at(0),v.at(1)});
    for(auto& v:j.at("tangents"))m.tangents.push_back({v.at(0),v.at(1),v.at(2),v.at(3)});
    for(auto& v:j.at("colors"))m.colors.push_back({v.at(0),v.at(1),v.at(2),v.at(3)});
    m.indices=j.at("indices").get<std::vector<uint32_t>>();m.materials=j.at("materials").get<std::vector<uint16_t>>();m.double_sided=j.at("double_sided").get<std::vector<uint8_t>>();
    if(auto e=blitz::validate(m.view());!e.empty())throw std::invalid_argument(e);return m;
}
inline std::string audit_hash(const nlohmann::json& j){
    auto bytes=j.dump();unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),digest);
    const char* hex="0123456789abcdef";std::string out;for(auto c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;
}
inline void save_audit_failure(const blitz::NeuralStats& stats,const std::filesystem::path& file){
    if(!stats.confirmation_failure||stats.confirmation_failure->reference.positions.empty())return;
    const auto& f=*stats.confirmation_failure;const auto& e=f.settings;auto a=audit_mesh(f.reference),b=audit_mesh(f.candidate);
    auto source=audit_mesh(f.source.positions.empty()?f.reference:f.source);
    nlohmann::json j={{"schema",2},{"stage",f.stage==blitz::NeuralAuditStage::PackingBaseline?"packing_baseline":"chain_confirmation"},{"raster",raster_name(f.raster)},{"storage",storage_name(f.storage)},
        {"source_sha256",audit_hash(source)},{"source",std::move(source)},
        {"backend",confirmation_name(f.backend)},{"reason",confirmation_reason(f.reason)},{"level",f.level},{"adjacent",f.adjacent},{"seconds",f.nanoseconds*1e-9},
        {"reference_sha256",audit_hash(a)},{"candidate_sha256",audit_hash(b)},{"reference",std::move(a)},{"candidate",std::move(b)},
        {"bounds",{f.bounds.center.x,f.bounds.center.y,f.bounds.center.z,f.bounds.radius}},
        {"settings",{{"screen_size",e.screen_size},{"limit",e.limit},{"max_changed_area",e.max_changed_area},{"profile",unsigned(e.profile)},
            {"weights",{e.weights.normal,e.weights.color,e.weights.material}},{"views",{e.views.orthographic,e.views.perspective,e.views.rotation_seed}},
            {"supersample",e.supersample},{"max_supersample",e.max_supersample},{"force_scalar",e.force_scalar},{"force_two_sided",e.force_two_sided}}},
        {"cpu",f.backend==blitz::NeuralConfirmation::Gpu?nlohmann::json(nullptr):audit_measurement(f.cpu)},
        {"gpu",f.backend==blitz::NeuralConfirmation::Cpu?nlohmann::json(nullptr):audit_measurement(f.gpu)}};
    std::filesystem::create_directories(file.parent_path());std::ofstream out(file);out<<j.dump(2)<<'\n';if(!out)throw std::runtime_error("failed to save audit replay");
}
inline nlohmann::json replay_audit(const std::filesystem::path& file,const blitz::NeuralOptions& options){
    nlohmann::json j;std::ifstream input(file);input>>j;auto version=j.at("schema").get<unsigned>();if(version!=1&&version!=2)throw std::invalid_argument("unsupported audit replay schema");
    if(audit_hash(j.at("reference"))!=j.at("reference_sha256").get<std::string>()||audit_hash(j.at("candidate"))!=j.at("candidate_sha256").get<std::string>())throw std::invalid_argument("audit replay mesh checksum mismatch");
    auto a=audit_mesh(j.at("reference")),b=audit_mesh(j.at("candidate"));const auto& s=j.at("settings");const auto& v=s.at("views"),w=s.at("weights"),box=j.at("bounds");
    blitz::Bounds bounds{{box.at(0),box.at(1),box.at(2)},box.at(3)};blitz::EvalSettings e;
    e.screen_size=s.at("screen_size");e.limit=s.at("limit");e.max_changed_area=s.at("max_changed_area");e.profile=blitz::Profile(s.at("profile").get<unsigned>());
    e.weights={w.at(0),w.at(1),w.at(2)};e.views={v.at(0),v.at(1),v.at(2)};e.supersample=s.at("supersample");e.max_supersample=s.at("max_supersample");e.force_scalar=s.at("force_scalar");e.force_two_sided=s.at("force_two_sided");
    auto controls=options;blitz::Mesh source;
    if(version==2){if(audit_hash(j.at("source"))!=j.at("source_sha256").get<std::string>())throw std::invalid_argument("audit replay source checksum mismatch");
        source=audit_mesh(j.at("source"));auto raster=j.at("raster").get<std::string>();
        if(raster=="vulkan-v1")controls.raster_backend=blitz::NeuralRasterBackend::Vulkan;else if(raster=="cuda-v1")controls.raster_backend=blitz::NeuralRasterBackend::Cuda;else throw std::invalid_argument("unsupported replay raster semantics");
        controls.vertex_storage=storage_option(j.at("storage").get<std::string>());}
    auto cpu=blitz::evaluate(a.view(),b.view(),bounds,e),gpu=version==1?blitz::evaluate_cuda(a.view(),b.view(),bounds,e,options):blitz::evaluate_gpu(a.view(),b.view(),bounds,e,controls,nullptr,source.view());
    bool agrees=cpu.passed==gpu.passed&&cpu.complete==gpu.complete&&cpu.resource_limited==gpu.resource_limited&&cpu.views_evaluated==gpu.views_evaluated;
    auto actual=audit_measurement(gpu);
    // JSON serializes infinite failed bounds as null. Compare the serialized
    // representation, rather than an in-memory infinity against parsed null.
    bool matches=!j.at("gpu").is_null()?j.at("gpu").dump()==actual.dump():!j.at("cpu").is_null()&&j.at("cpu").dump()==audit_measurement(cpu).dump();
    return {{"decisions_agree",agrees},{"recorded_matches",matches},{"recorded_gpu_matches",!j.at("gpu").is_null()&&matches},{"reproduced",version==1?agrees:matches},{"raster",version==1?"cuda-v1":raster_name(controls.raster_backend)},{"cpu",audit_measurement(cpu)},{"gpu",std::move(actual)}};
}
