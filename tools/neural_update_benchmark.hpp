#pragma once
#include "neural_resident.hpp"
namespace blitz::neural::training {
inline void benchmark_updates(const fs::path& shard,const fs::path& output,const fs::path& weights){
    if(fs::exists(output))throw std::invalid_argument("choose a fresh update benchmark report");
    auto initial=load_weights(weights);if(initial.architecture!=placement_schema||initial.hidden_width!=64)throw std::invalid_argument("update benchmark needs a width-64 placement policy");torch::Device device(torch::kCUDA);auto index=read_json(shard/"index.json");auto dataset=load_actions(shard/"actions.bin");ResidentDataset data(device,519,true);data.append(dataset,index.at("asset"),index.at("category"));
    json report={{"complete",false},{"score",nullptr},{"updates",2048},{"batch",512},{"input_sha256",file_sha256(shard/"actions.bin")},{"binary_sha256",file_sha256("/proc/self/exe")},{"model_sha256",file_sha256(weights)},{"rows",json::array()}};
    for(unsigned repeat=0;repeat<5;++repeat)for(unsigned variant=0;variant<2;++variant){auto backend=(variant^(repeat&1))?UpdateBackend::Fused:UpdateBackend::Reference;
        ActionNetwork model(placement_schema);model->to(device);{torch::NoGradGuard guard;size_t at=0;for(auto& p:model->parameters()){p.copy_(torch::from_blob(initial.values.data()+at,p.sizes(),torch::kFloat32));at+=p.numel();}}
        ResidentUpdate update(model,data,512,{},backend);auto saved=update.optimizer.snapshot();update.run(64);update.optimizer.restore(saved);cuda_check(cudaDeviceSynchronize());
        cudaEvent_t start{},finish{};cuda_check(cudaEventCreate(&start));cuda_check(cudaEventCreate(&finish));auto begin=std::chrono::steady_clock::now();cuda_check(cudaEventRecord(start));update.run(2048);cuda_check(cudaEventRecord(finish));cuda_check(cudaEventSynchronize(finish));auto wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();float ms=0;cuda_check(cudaEventElapsedTime(&ms,start,finish));cuda_check(cudaEventDestroy(start));cuda_check(cudaEventDestroy(finish));
        auto state=update.optimizer.state();if(state.failure||state.step!=2048)throw std::runtime_error("update benchmark failed");report["rows"].push_back({{"backend",backend==UpdateBackend::Fused?"fused":"reference"},{"repeat",repeat},{"seconds",wall},{"gpu_seconds",ms*.001},{"updates_per_second",2048/wall},{"epilogues",update.fused_epilogues()},{"loss",state.loss},{"gradient",state.gradient},{"pool",update.pool()}});write_json(output,report);
    }
    report["complete"]=true;write_json(output,report);
}
}
