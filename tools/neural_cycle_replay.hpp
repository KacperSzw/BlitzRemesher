#pragma once
#include "neural_resident.hpp"
#include "neural_checkpoint.hpp"

namespace blitz::neural::training {
// A forensic continuation has its own provenance and output directory. It never
// rewrites the source run or pretends a different build/device is an exact resume.
inline int replay_cycle_checkpoint(const fs::path& source,const fs::path& output,uint32_t updates){
    if(!updates||updates>100000||fs::exists(output))throw std::invalid_argument("replay requires 1..100000 updates and a new output directory");
    const auto journal=read_json(source/"latest.json"),contract=read_json(source/"contract.json");
    const auto checkpoint=source/journal.at("checkpoint").get<std::string>();
    if(file_sha256(checkpoint/"checkpoint.pt")!=journal.at("checkpoint_sha256").get<std::string>())throw std::invalid_argument("replay checkpoint checksum differs");
    fs::create_directories(output);torch::Device device(torch::kCUDA);
    ResidentDataset data(device,contract.at("seed"),contract.at("data_storage")=="compact-v1");
    for(size_t i=journal.value("resident_start",size_t(0));i<journal.at("datasets").size();++i){
        auto directory=source/"data"/journal.at("datasets")[i].get<std::string>();auto index=read_json(directory/"index.json");
        if(!index.at("complete").get<bool>()||!index.at("reference_confirmed").get<bool>()||index.at("sha256")!=file_sha256(directory/"actions.bin")||index.at("contract_sha256")!=file_sha256(directory/"contract.json"))throw std::invalid_argument("replay dataset checksum differs");
        if(contract.at("data_storage")=="compact-v1")data.append(load_compact_actions(directory/"actions.bin"),index.at("asset"),index.at("category"));
        else data.append(load_actions(directory/"actions.bin"),index.at("asset"),index.at("category"));
    }
    ActionNetwork model(placement_schema);model->to(device);
    ResidentUpdate update(model,data,contract.at("batch"),{},contract.at("update_backend")=="reference"?UpdateBackend::Reference:UpdateBackend::Fused);
    const auto initial=load_state(checkpoint/"checkpoint.pt",model,update.optimizer,device);
    const auto target=update_target(uint32_t(initial),updates);cudaDeviceProp properties{};cuda_check(cudaGetDeviceProperties(&properties,0));
    json report={{"source_checkpoint_sha256",journal.at("checkpoint_sha256")},{"source_contract",contract},{"binary_sha256",file_sha256("/proc/self/exe")},{"device",properties.name},{"initial_step",initial},{"target_step",target},{"states",data.states()},{"complete",false},{"checks",json::array()}};
    CheckpointWriter writer(output/"scratch");
    while(update.optimizer.state().step<target){
        auto state=update.optimizer.state();update.run(std::min(128u,target-state.step));state=update.optimizer.state();
        torch::NoGradGuard guard;auto input=update.input.slice(0,0,std::min<int64_t>(8,contract.at("batch").get<int64_t>())).flatten(0,1).contiguous();
        auto expected=model->forward(input);auto host=input.cpu().contiguous();auto weights=export_actions(model,{{"step",state.step}});
        auto reference=action_oracle({host.data_ptr<float>(),size_t(host.numel())},weights);auto difference=numeric_difference(doubles(expected),reference);
        json check={{"step",state.step},{"loss",state.loss},{"gradient",state.gradient},{"optimizer_failure",state.failure},{"fp64",difference_json(difference)},{"input_max_abs",host.abs().max().item<float>()}};
        if(state.failure||!legacy_numeric_pass(difference)){
            auto failed=output/("step-"+std::to_string(state.step));fs::create_directories(failed);save_state(failed/"checkpoint.pt",model,update.optimizer,state.step);save_weights(failed/"model.blzn",weights);
            torch::save(host,failed/"input.pt");torch::save(expected.cpu(),failed/"expected.pt");torch::save(update.ids.cpu(),failed/"sample_ids.pt");
            ActionCuda native(weights,{},32);auto values=native.predict({host.data_ptr<float>(),size_t(host.numel())});std::vector<double> actual(values.begin(),values.end());
            check["native_fp64"]=difference_json(numeric_difference(actual,reference));check["torch_native"]=difference_json(numeric_difference(doubles(expected),actual));
            check["weights_max_abs"]=*std::max_element(weights.values.begin(),weights.values.end(),[](float a,float b){return std::abs(a)<std::abs(b);});
            write_json(failed/"verification.json",check);report["checks"].push_back(check);report["failure"]=failed.filename().string();write_json(output/"report.json",report);std::cout<<check.dump()<<'\n';return 2;
        }
        report["checks"].push_back(check);write_json(output/"report.json",report);
        if(state.step%512==0)writer.submit(output/"checkpoints"/("step-"+std::to_string(state.step)),model,update.optimizer,{{"forensic",true}},input,expected,output/"latest.json",{{"forensic",true}});
    }
    writer.join();report["complete"]=true;write_json(output/"report.json",report);std::cout<<json({{"replay_complete",true},{"initial_step",initial},{"step",target}}).dump()<<'\n';return 0;
}
}
