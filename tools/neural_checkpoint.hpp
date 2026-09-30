#pragma once
#include "neural_update.hpp"
#include <future>
namespace blitz::neural::training {
inline void restore_model(torch::serialize::InputArchive& archive,ActionNetwork& model,torch::Device device){
    ActionNetwork restored(model->architecture);restored->to(device);restored->load(archive);torch::NoGradGuard guard;auto current=model->parameters(),loaded=restored->parameters();
    for(size_t i=0;i<current.size();++i){if(current[i].sizes()!=loaded[i].sizes())throw std::invalid_argument("checkpoint model dimensions");current[i].copy_(loaded[i]);}
}
inline void load_checkpoint_model(const fs::path& path,ActionNetwork& model,torch::Device device){
    torch::serialize::InputArchive all,m;all.load_from(path.string(),device);all.read("model",m);restore_model(m,model,device);
}
inline void save_state(const fs::path& path,ActionNetwork& model,DeviceAdam& optimizer,uint64_t step){
    torch::serialize::OutputArchive all,m,o;model->save(m);optimizer.save(o);all.write("model",m);all.write("optimizer",o);all.write("step",torch::tensor(int64_t(step)));auto temp=path;temp+=".part";all.save_to(temp.string());fs::rename(temp,path);
}
inline uint64_t load_state(const fs::path& path,ActionNetwork& model,DeviceAdam& optimizer,torch::Device device){
    torch::serialize::InputArchive all,m,o;all.load_from(path.string(),device);all.read("model",m);all.read("optimizer",o);restore_model(m,model,device);optimizer.load(o);torch::Tensor step;all.read("step",step);auto n=step.item<int64_t>();if(n<0||uint64_t(n)!=optimizer.state().step)throw std::invalid_argument("checkpoint counter mismatch");return uint64_t(n);
}
// One frozen CPU snapshot in flight. This thread never touches live GPU model
// or optimizer tensors, and destruction joins outstanding I/O.
class CheckpointWriter {
    std::future<void> pending_;
public:
    ~CheckpointWriter(){if(pending_.valid())pending_.wait();}
    void join(){if(pending_.valid())pending_.get();}
    void submit(const fs::path& path,ActionNetwork& model,DeviceAdam& optimizer,const json& provenance,torch::Tensor input={},torch::Tensor expected={},fs::path journal_path={},json journal={}){
        join();torch::NoGradGuard guard;ActionNetwork frozen(model->architecture);auto source=model->parameters(),target=frozen->parameters();for(size_t i=0;i<source.size();++i)target[i].copy_(source[i].detach().cpu());
        auto control=optimizer.control.cpu().clone();std::vector<torch::Tensor> mean,variance;for(auto& v:optimizer.mean)mean.push_back(v.cpu().clone());for(auto& v:optimizer.variance)variance.push_back(v.cpu().clone());auto step=optimizer.state().step;
        if(input.defined())input=input.detach().cpu().clone();if(expected.defined())expected=expected.detach().cpu().clone();
        pending_=std::async(std::launch::async,[path,frozen=std::move(frozen),control=std::move(control),mean=std::move(mean),variance=std::move(variance),step,provenance,input=std::move(input),expected=std::move(expected),journal_path=std::move(journal_path),journal=std::move(journal)]()mutable{
            torch::NoGradGuard guard;fs::create_directories(path);torch::serialize::OutputArchive all,m,o;frozen->save(m);o.write("version",torch::tensor(int64_t(update_checkpoint_version)));o.write("sampler",torch::tensor(int64_t(sampler_version)));o.write("control",control);
            for(size_t i=0;i<mean.size();++i){o.write("mean"+std::to_string(i),mean[i]);o.write("variance"+std::to_string(i),variance[i]);}all.write("model",m);all.write("optimizer",o);all.write("step",torch::tensor(int64_t(step)));all.save_to((path/"checkpoint.pt.part").string());fs::rename(path/"checkpoint.pt.part",path/"checkpoint.pt");
            auto weights=export_actions(frozen,provenance);save_weights(path/"model.blzn",weights);
            if(input.defined()){torch::save(input,path/"input.pt");torch::save(expected,path/"expected.pt");auto flat=input.contiguous();auto reference=action_oracle({flat.data_ptr<float>(),size_t(flat.numel())},weights);auto difference=numeric_difference(doubles(expected),reference);write_json(path/"verification.json",{{"fp64",difference_json(difference)}});if(!legacy_numeric_pass(difference))throw std::runtime_error("checkpoint FP64 export verification failed");}
            auto checksum=file_sha256(path/"checkpoint.pt");write_json(path/"index.json",{{"complete",true},{"step",step},{"checkpoint_sha256",checksum},{"model_sha256",file_sha256(path/"model.blzn")}});
            // Publish the recovery pointer only after every frozen artifact and
            // numerical check has succeeded. A crash leaves the prior pointer.
            if(!journal_path.empty()){journal["step"]=step;journal["checkpoint"]=fs::relative(path,journal_path.parent_path()).string();journal["checkpoint_sha256"]=checksum;write_json(journal_path,journal);}
        });
    }
};
}
