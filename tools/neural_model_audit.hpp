#pragma once
#include "neural_json.hpp"
#include "neural_data.hpp"
#include "neural_cuda.cuh"
namespace blitz::neural::training {
inline void audit_model(const fs::path& manifest_path,const fs::path& model_path,const fs::path& settings_path,const fs::path& output){
    if(fs::exists(output))throw std::invalid_argument("model audit output exists");auto manifest=read_json(manifest_path),config=read_json(settings_path);auto settings=settings_json(config);NeuralOptions options;options.raster_backend=NeuralRasterBackend::Vulkan;options.memory_mib=config.value("gpu_memory_mib",4096u);options.action_trials=8;options.action_batch=16;
    bool complete=true;json report={{"complete",false},{"score",nullptr},{"gpu_memory_mib",options.memory_mib},{"model_sha256",file_sha256(model_path)},{"manifest_sha256",file_sha256(manifest_path)},{"settings_sha256",file_sha256(settings_path)},{"rows",json::array()}};auto start=std::chrono::steady_clock::now();
    gpu::StreamScope stream;MemoryScope memory(options);AuditSession session(options);
    for(const auto& asset:manifest.at("assets")){
        if(asset.at("split")!="development"&&asset.at("split")!="validation")throw std::invalid_argument("model comparison cannot tune on release holdout");
        for(const auto& f:asset.at("files"))if(f.at("sha256")!=file_sha256(f.at("path").get<std::string>()))throw std::invalid_argument("model audit source changed");auto mesh=load_mesh(asset.at("path").get<std::string>());
        for(auto ranking:{NeuralRanking::Constant,NeuralRanking::Learned}){options.ranking=ranking;NeuralModel model(model_path.c_str(),options);NeuralStats stats;json row;auto began=std::chrono::steady_clock::now();memory.budget.peak.store(memory.budget.live.load());
            try{auto result=generate_neural(mesh.view(),settings,model,&stats);row=result_json(result);if(result.status!=Status::Complete||stats.resource_failures||stats.confirmation_nonfinite||stats.confirmation_cancelled)complete=false;}
            catch(const gpu::ResourceError& e){row={{"status","failed"},{"error",e.what()},{"workspace_limited",e.kind==NeuralResourceLimit::WorkspaceMemory},{"requested_bytes",e.requested},{"limit_bytes",e.limit}};complete=false;}
            catch(const std::exception& e){row={{"status","failed"},{"error",e.what()}};complete=false;}
            row["asset"]=asset.at("id");row["ranking"]=ranking_name(ranking);row["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();row["gpu_workspace_peak_bytes"]=memory.budget.peak.load();row["neural"]=neural_json(stats);report["rows"].push_back(row);write_json(output,report);}
    }
    report["complete"]=complete;report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();write_json(output,report);
}
}
