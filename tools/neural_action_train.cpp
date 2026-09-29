#include "neural_action_network.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <random>
#include <iostream>
#include <csignal>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
struct Dataset {std::vector<float> x;std::vector<uint8_t> labels,valid;std::vector<std::array<std::vector<uint32_t>,4>> asset_bins;std::vector<std::vector<uint32_t>> categories;json provenance=json::array();uint32_t states{};};
Dataset dataset(const fs::path& directory) {
    Dataset out;auto index=read_json(directory/"index.json");std::vector<fs::path> paths{directory};
    if(index.contains("datasets")){paths.clear();for(auto& p:index.at("datasets")){fs::path relative=p.get<std::string>();if(relative.is_absolute()||relative.string().find("..")!=std::string::npos)throw std::invalid_argument("invalid dataset path");paths.push_back(directory/relative);}}
    std::vector<std::string> names,asset_names,asset_categories;
    for(auto& path:paths){auto j=read_json(path/"index.json");if(!j.at("complete").get<bool>()||j.at("schema")!=action_schema||j.at("path")!="actions.bin"||j.at("sha256")!=file_sha256(path/"actions.bin")||j.at("contract_sha256")!=file_sha256(path/"contract.json"))throw std::invalid_argument("incomplete or changed action dataset");
        auto data=load_actions(path/"actions.bin");if(!data.states())throw std::invalid_argument("action dataset has no states");out.provenance.push_back({{"asset",j.at("asset")},{"sha256",j.at("sha256")},{"contract_sha256",j.at("contract_sha256")}});
        if(uint64_t(out.states)+data.states()>(8ull<<30)/(action_pool*action_features*sizeof(float)))throw std::length_error("resident action dataset exceeds 8 GiB");
        const auto category=j.at("category").get<std::string>();auto found=std::find(names.begin(),names.end(),category);if(found==names.end()){names.push_back(category);out.categories.emplace_back();found=std::prev(names.end());}
        auto asset=j.at("asset").get<std::string>();auto asset_it=std::find(asset_names.begin(),asset_names.end(),asset);size_t asset_id=size_t(asset_it-asset_names.begin());
        if(asset_it==asset_names.end()){asset_names.push_back(asset);asset_categories.push_back(category);out.categories[size_t(found-names.begin())].push_back(uint32_t(asset_id));out.asset_bins.emplace_back();}
        else if(asset_categories[asset_id]!=category)throw std::invalid_argument("asset appears in multiple categories");
        for(uint32_t s=0;s<data.states();++s){auto count=data.offsets[s+1]-data.offsets[s];out.asset_bins[asset_id][std::min(3u,uint32_t(data.progress[s]*4))].push_back(out.states++);
            out.x.insert(out.x.end(),data.x.begin()+size_t(data.offsets[s])*action_features,data.x.begin()+size_t(data.offsets[s+1])*action_features);out.x.resize(size_t(out.states)*action_pool*action_features);
            out.labels.insert(out.labels.end(),data.labels.begin()+data.offsets[s],data.labels.begin()+data.offsets[s+1]);out.labels.resize(size_t(out.states)*action_pool);
            out.valid.insert(out.valid.end(),count,1);out.valid.resize(size_t(out.states)*action_pool);}
    }
    if(out.x.size()*sizeof(float)>8ull*1024*1024*1024)throw std::length_error("resident action dataset exceeds 8 GiB");return out;
}
void save_state(const fs::path& path,ActionNetwork& model,torch::optim::AdamW& optimizer,uint64_t step) {
    torch::serialize::OutputArchive all,m,o;model->save(m);optimizer.save(o);all.write("model",m);all.write("optimizer",o);all.write("step",torch::tensor(int64_t(step)));auto temp=path;temp+=".part";all.save_to(temp.string());fs::rename(temp,path);
}
uint64_t load_state(const fs::path& path,ActionNetwork& model,torch::optim::AdamW& optimizer,torch::Device device) {
    torch::serialize::InputArchive all,m,o;all.load_from(path.string(),device);all.read("model",m);all.read("optimizer",o);model->load(m);optimizer.load(o);torch::Tensor step;all.read("step",step);return step.item<int64_t>();
}
void check_contracts(const fs::path& output={}) {
    torch::manual_seed(771);torch::Device device(torch::kCUDA);ActionNetwork model;model->to(device);auto input=torch::randn({35,action_features},torch::TensorOptions().device(device));auto expected=model->forward(input);
    auto w=export_actions(model,{{"test",true}});ActionCuda native(w,{},7);auto host=input.cpu().contiguous();std::span<const float> x{host.data_ptr<float>(),size_t(host.numel())};auto result=native.predict(x);std::vector<double> actual(result.begin(),result.end());
    auto d=numeric_difference(doubles(expected),actual);if(!legacy_numeric_pass(d)||!legacy_numeric_pass(numeric_difference(actual,action_oracle(x,w))))throw std::runtime_error("action native/FP64 parity failed");
    auto prediction=torch::zeros({2,4,3},torch::TensorOptions().device(device).requires_grad(true));
    auto labels=torch::tensor({15,11,8,0,15,15,11,0},torch::kUInt8).view({2,4}).to(device),valid=labels.bitwise_and(Queried).ne(0);
    auto loss=action_loss(prediction,labels,valid);loss.backward();auto grad=prediction.grad().cpu();
    if(grad[0][0][0].item<float>()>=0||grad[0][1][0].item<float>()<=0||grad.select(1,3).abs().sum().item<float>()!=0)throw std::runtime_error("action preferred/unknown loss contract");
    auto permutation=torch::tensor({1,0,2,3},torch::kInt64).to(device);if(std::abs(action_loss(prediction.index_select(1,permutation),labels.index_select(1,permutation),valid.index_select(1,permutation)).item<double>()-loss.item<double>())>1e-6)throw std::runtime_error("action loss depends on candidate ordering");
    if(!output.empty())save_weights(output,w);
    std::cout<<json({{"training_started",false},{"native_difference",difference_json(d)},{"loss_contracts",true}}).dump()<<'\n';
}
int main(int argc,char** argv){try{
    ieee_fp32();torch::set_num_threads(4);if((argc==2||argc==3)&&std::string_view(argv[1])=="--check"){if(!torch::cuda::is_available())return 77;check_contracts(argc==3?fs::path(argv[2]):fs::path{});return 0;}
    if(argc==4&&std::string_view(argv[1])=="--replay"){
        fs::path bundle=argv[2];auto w=load_weights(bundle/"model.blzn");torch::Tensor input,expected;torch::load(input,(bundle/"input.pt").string());torch::load(expected,(bundle/"expected.pt").string());
        input=input.to(torch::kCPU,torch::kFloat32).contiguous();if(input.numel()>32768*action_features||input.dim()!=2||input.size(1)!=action_features)throw std::invalid_argument("invalid replay tensor shape/cap");
        std::span<const float> features{input.data_ptr<float>(),size_t(input.numel())};ActionCuda native(w,{});auto values=native.predict(features);std::vector<double> actual(values.begin(),values.end());
        auto difference=numeric_difference(doubles(expected),actual),oracle=numeric_difference(actual,action_oracle(features,w));
        auto passed=legacy_numeric_pass(difference)&&legacy_numeric_pass(oracle);json report={{"training_started",false},{"model_sha256",file_sha256(bundle/"model.blzn")},{"input_sha256",file_sha256(bundle/"input.pt")},{"expected_sha256",file_sha256(bundle/"expected.pt")},{"native",difference_json(difference)},{"fp64",difference_json(oracle)},{"passed",passed}};
        write_json(argv[3],report);std::cout<<report.dump(2)<<'\n';return passed?0:1;
    }
    if(argc<3)throw std::invalid_argument("blitz-neural-action-train DATASET RUN [--steps N] [--minutes N] [--batch N] [--seed N]");
    uint64_t steps=10000,seed=0xB1172026;uint32_t batch=512;double minutes=5;
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing action trainer option");std::string k=argv[i];if(k=="--steps")steps=std::stoull(argv[i+1]);else if(k=="--minutes")minutes=std::stod(argv[i+1]);else if(k=="--batch")batch=uint32_t(std::stoul(argv[i+1]));else if(k=="--seed")seed=std::stoull(argv[i+1]);else throw std::invalid_argument("unknown action trainer option");}
    if(!steps||steps>1000000||!batch||batch>4096||!std::isfinite(minutes)||minutes<=0||minutes>50||seed>UINT32_MAX)throw std::invalid_argument("action trainer bounds");
    if(!torch::cuda::is_available())throw NeuralUnavailable("action training requires CUDA");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);torch::manual_seed(seed);torch::Device device(torch::kCUDA);size_t free_bytes=0,total_bytes=0;
    if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)throw std::runtime_error("CUDA memory query failed");c10::cuda::CUDACachingAllocator::setMemoryFraction(std::min(.5,double(12ull<<30)/total_bytes),0);
    fs::path directory=argv[1],run=argv[2];fs::create_directories(run);auto data=dataset(directory);
    json contract={{"schema",action_schema},{"data",data.provenance},{"seed",seed},{"batch",batch},{"precision","IEEE FP32"},{"margin",1},{"auxiliary",.25},{"logit_penalty",1e-4},{"lr",.001},{"weight_decay",.0001},{"libtorch",TORCH_VERSION},{"binary_sha256",file_sha256("/proc/self/exe")}};
    if(fs::exists(run/"contract.json")&&read_json(run/"contract.json")!=contract)throw std::invalid_argument("action training contract changed");write_json(run/"contract.json",contract);
    auto x=torch::from_blob(data.x.data(),{data.states,action_pool,action_features},torch::kFloat32).clone().to(device);
    auto labels=torch::from_blob(data.labels.data(),{data.states,action_pool},torch::kUInt8).clone().to(device),valid=torch::from_blob(data.valid.data(),{data.states,action_pool},torch::kUInt8).clone().to(device).to(torch::kBool);
    ActionNetwork model;model->to(device);torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));uint64_t step=0;
    if(fs::exists(run/"latest.json")){auto latest=read_json(run/"latest.json");auto file=run/latest.at("checkpoint").get<std::string>();if(latest.at("checkpoint_sha256")!=file_sha256(file))throw std::invalid_argument("action checkpoint changed");step=load_state(file,model,optimizer,device);if(step!=latest.at("step").get<uint64_t>())throw std::invalid_argument("action checkpoint step mismatch");}
    auto start=std::chrono::steady_clock::now();auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto initial=model->parameters().front().detach().clone();uint64_t began=step;double last_loss=0,first_loss=0,gradient=0;
    std::ofstream log(run/"metrics.jsonl",std::ios::app);std::vector<int64_t> ids(batch);
    auto checkpoint=[&]{torch::NoGradGuard guard;auto name="step-"+std::to_string(step);auto bundle=run/"forensic"/name;fs::create_directories(bundle);
        auto w=export_actions(model,{{"schema",action_schema},{"step",step},{"contract",file_sha256(run/"contract.json")}});save_state(bundle/"checkpoint.pt",model,optimizer,step);save_weights(bundle/"model.blzn",w);
        auto input=x.slice(0,0,std::min<int64_t>(data.states,32)).flatten(0,1);torch::save(input.cpu(),bundle/"input.pt");auto expected=model->forward(input);torch::save(expected.cpu(),bundle/"expected.pt");
        auto cpu=input.cpu().contiguous();std::span<const float> features{cpu.data_ptr<float>(),size_t(cpu.numel())};ActionCuda native(w,{});auto predicted=native.predict(features);std::vector<double> actual(predicted.begin(),predicted.end());
        torch::save(torch::from_blob(predicted.data(),{input.size(0),action_outputs},torch::kFloat32).clone(),bundle/"native.pt");
        auto difference=numeric_difference(doubles(expected),actual),oracle=numeric_difference(actual,action_oracle(features,w));
        write_json(bundle/"verification.json",{{"native",difference_json(difference)},{"fp64",difference_json(oracle)},{"threshold",2e-4}});
        if(!legacy_numeric_pass(difference)||!legacy_numeric_pass(oracle))throw std::runtime_error("action export parity failed; forensic bundle retained");
        ActionNetwork restored;restored->to(device);torch::optim::AdamW restored_optimizer(restored->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));if(load_state(bundle/"checkpoint.pt",restored,restored_optimizer,device)!=step)throw std::runtime_error("action restore step mismatch");
        for(size_t i=0;i<model->parameters().size();++i){auto a=model->parameters()[i],b=restored->parameters()[i];if(!torch::equal(a,b))throw std::runtime_error("action restore parameters mismatch");
            auto& before=static_cast<torch::optim::AdamWParamState&>(*optimizer.state().at(a.unsafeGetTensorImpl()));auto& after=static_cast<torch::optim::AdamWParamState&>(*restored_optimizer.state().at(b.unsafeGetTensorImpl()));
            if(before.step()!=after.step()||!torch::equal(before.exp_avg(),after.exp_avg())||!torch::equal(before.exp_avg_sq(),after.exp_avg_sq()))throw std::runtime_error("action restore optimizer mismatch");}
        double changed=(initial-model->parameters().front()).abs().max().item<double>();if(!std::isfinite(changed)||changed<=0)throw std::runtime_error("action parameters did not update");
        double correct=0,eligible=0;for(uint32_t s=0;s<data.states;s+=1024){auto end=std::min(data.states,s+1024);auto y=labels.slice(0,s,end),mask=valid.slice(0,s,end);double count=y.bitwise_and(Preferred).ne(0).logical_and(mask).any(1).sum().item<double>();correct+=action_membership(model->forward(x.slice(0,s,end)),y,mask)*count;eligible+=count;}
        auto path=run/(name+".pt");fs::rename(bundle/"checkpoint.pt",path);fs::rename(bundle/"model.blzn",run/(name+".blzn"));
        for(auto file:{"input.pt","expected.pt","native.pt","verification.json"})fs::remove(bundle/file);fs::remove(bundle);
        json health={{"step",step},{"checkpoint",name+".pt"},{"checkpoint_sha256",file_sha256(path)},{"model",name+".blzn"},{"model_sha256",file_sha256(run/(name+".blzn"))},{"native_max_abs",difference.maximum},{"fp64_max_abs",oracle.maximum},{"restored",true},{"optimizer_restored",true},{"finite",true},{"parameter_change",changed},{"preferred_membership",eligible?correct/eligible:0},{"preferred_states",eligible},{"states",data.states},{"first_loss",first_loss},{"last_loss",last_loss},{"gradient_norm",gradient},{"seconds",elapsed()},{"steps_this_segment",step-began},{"complete",step>=steps},{"proof_passed",false}};
        write_json(run/"latest.json",health);std::cout<<health.dump()<<std::endl;
    };
    while(step<steps&&elapsed()<minutes*60&&!stopped){std::mt19937 rng(uint32_t(seed)^uint32_t(step*0x9e3779b9));
        for(auto& id:ids){auto& category=data.categories[rng()%data.categories.size()];auto& bins=data.asset_bins[category[rng()%category.size()]];std::array<unsigned,4> available{};unsigned count=0;for(unsigned i=0;i<4;++i)if(!bins[i].empty())available[count++]=i;auto& bin=bins[available[rng()%count]];id=bin[rng()%bin.size()];}
        auto indices=torch::from_blob(ids.data(),{batch},torch::kInt64).clone().to(device);auto input=x.index_select(0,indices),target=labels.index_select(0,indices),mask=valid.index_select(0,indices);
        optimizer.zero_grad();auto prediction=model->forward(input);auto loss=action_loss(prediction,target,mask);
        auto failure=[&](const char* reason){auto bundle=run/"forensic"/("update-"+std::to_string(step));fs::create_directories(bundle);
            write_json(bundle/"failure.json",{{"complete",false},{"reason",reason},{"step_before_update",step},{"sample_ids",ids}});save_state(bundle/"checkpoint.pt",model,optimizer,step);
            torch::save(input.flatten(0,1).cpu(),bundle/"input.pt");torch::save(prediction.flatten(0,1).cpu(),bundle/"expected.pt");torch::save(target.cpu(),bundle/"labels.pt");torch::save(mask.cpu(),bundle/"valid.pt");
            auto w=export_actions(model,{{"failure",reason},{"step",step}});save_weights(bundle/"model.blzn",w);write_json(bundle/"failure.json",{{"complete",true},{"reason",reason},{"step_before_update",step},{"sample_ids",ids}});
            throw std::runtime_error(std::string(reason)+"; forensic bundle retained");};
        last_loss=loss.item<double>();if(!std::isfinite(last_loss))failure("nonfinite action loss");if(step==began)first_loss=last_loss;
        loss.backward();gradient=torch::nn::utils::clip_grad_norm_(model->parameters(),1);if(!std::isfinite(gradient))failure("nonfinite action gradient");optimizer.step();++step;
        if(step%128==0){log<<json({{"step",step},{"loss",last_loss},{"gradient",gradient},{"seconds",elapsed()}}).dump()<<'\n';log.flush();}
        if(step%512==0)checkpoint();
    }
    if(step>began&&step%512)checkpoint();return step>=steps?0:2;
}catch(const std::exception& e){std::cerr<<"action training: "<<e.what()<<'\n';return 1;}}
