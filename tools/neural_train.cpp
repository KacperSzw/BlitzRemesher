#include "neural_data.hpp"
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <random>
#include <numeric>
#include <iostream>
#include <csignal>
#include <condition_variable>
#include <thread>
#include <string_view>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
using Clock=std::chrono::steady_clock;
struct NetworkImpl:torch::nn::Module {
    std::vector<torch::nn::Linear> layer;
    NetworkImpl(){for(int i=0;i<5;++i)layer.push_back(register_module("layer"+std::to_string(i),torch::nn::Linear(layer_in[i],layer_out[i])));}
    torch::Tensor encode(torch::Tensor x,const torch::Tensor& from,const torch::Tensor& to,const torch::Tensor& degree){
        for(unsigned i=0;i<3;++i){auto mean=torch::zeros_like(x);mean.index_add_(0,from,x.index_select(0,to));x=torch::relu(layer[i]->forward(torch::cat({x,mean/degree},1)));}return x;
    }
    torch::Tensor head(torch::Tensor x,const torch::Tensor& c){return layer[4]->forward(torch::relu(layer[3]->forward(torch::cat({x,c},1))));}
};TORCH_MODULE(Network);
struct Batch {
    torch::Tensor x,from,to,degree,conditions,core,target;
    Patch first;std::array<float,neural::conditions> first_condition{};
    double preparation_seconds=0;
    void upload(torch::Device device){for(auto* value:{&x,&from,&to,&degree,&conditions,&core,&target})*value=value->to(device,true);}
};
template<class T> torch::Tensor tensor(const std::vector<T>& data,std::vector<int64_t> shape,torch::ScalarType dtype,torch::Device device) {
    auto pinned=torch::empty(shape,torch::TensorOptions().dtype(dtype).pinned_memory(true));std::memcpy(pinned.data_ptr(),data.data(),data.size()*sizeof(T));return pinned.to(device,true);
}
Batch make_batch(const std::vector<Asset>& assets,uint64_t step,uint32_t core_count,uint32_t count,torch::Device device) {
    auto begin=Clock::now();std::mt19937 rng(uint32_t(0xB1172026u+step*2654435761u));Batch batch;
    std::vector<float> x,degree,cs,targets;std::vector<int64_t> from,to,core;
    for(unsigned item=0;item<count;++item) {
        auto& asset=assets[rng()%assets.size()];auto& g=asset.graph;auto& e=asset.examples[rng()%asset.examples.size()];
        std::vector<uint32_t> ids;std::vector<uint8_t> seen(g.size());uint32_t next=rng()%g.size();
        // Connected cores limit halo duplication. Isolated/small components fill deterministically.
        auto add=[&](uint32_t id){if(!seen[id]&&ids.size()<core_count){seen[id]=1;ids.push_back(id);}};add(next);
        size_t cursor=0;while(ids.size()<std::min<size_t>(core_count,g.size())) {
            if(cursor<ids.size()){auto v=ids[cursor++];for(auto k=g.offsets[v];k<g.offsets[v+1];++k)add(g.neighbors[k]);}
            else {while(seen[next])next=(next+1)%uint32_t(g.size());add(next);}
        }
        Patch p;for(;;){try{p=patch(g,ids,32768);break;}catch(const std::length_error&){if(ids.size()<2)throw;ids.resize(ids.size()/2);}}
        size_t base=x.size()/features;x.insert(x.end(),p.graph.x.begin(),p.graph.x.end());
        for(uint32_t i=0;i<p.graph.size();++i){cs.insert(cs.end(),e.condition.begin(),e.condition.end());degree.push_back(float(std::max(1u,p.graph.offsets[i+1]-p.graph.offsets[i])));
            for(auto k=p.graph.offsets[i];k<p.graph.offsets[i+1];++k){from.push_back(int64_t(base+i));to.push_back(int64_t(base+p.graph.neighbors[k]));}}
        for(uint32_t i=0;i<p.core;++i){auto id=p.ids[i],r=e.representative[id];core.push_back(int64_t(base+i));targets.push_back(float(e.retained[id]));
            for(unsigned k=0;k<3;++k)targets.push_back(g.x[size_t(r)*features+k]-g.x[size_t(id)*features+k]);}
        if(!item){batch.first=std::move(p);batch.first_condition=e.condition;}
    }
    batch.x=tensor(x,{int64_t(x.size()/features),features},torch::kFloat32,device);batch.from=tensor(from,{int64_t(from.size())},torch::kInt64,device);batch.to=tensor(to,{int64_t(to.size())},torch::kInt64,device);
    batch.degree=tensor(degree,{int64_t(degree.size()),1},torch::kFloat32,device);batch.conditions=tensor(cs,{int64_t(degree.size()),neural::conditions},torch::kFloat32,device);
    batch.core=tensor(core,{int64_t(core.size())},torch::kInt64,device);batch.target=tensor(targets,{int64_t(core.size()),outputs},torch::kFloat32,device);
    batch.preparation_seconds=std::chrono::duration<double>(Clock::now()-begin).count();return batch;
}
// Four pinned host batches bound storage. Up to four CPU packers supply CUDA.
// Slots are consumed in step order regardless of completion order, preserving sampling.
// Workers own no model/device tensors; asset storage outlives their joined threads.
class Prefetch {
    std::mutex mutex;std::condition_variable ready;std::array<std::optional<Batch>,4> slots;
    uint64_t producing,consuming;bool stopping=false;std::exception_ptr failure;std::vector<std::thread> workers;
public:
    Prefetch(const std::vector<Asset>& assets,uint64_t first,uint64_t end,uint32_t core,uint32_t count,uint32_t worker_count):producing(first),consuming(first){
        try{for(unsigned i=0;i<worker_count;++i)workers.emplace_back([&,end,core,count]{
            try{for(;;){uint64_t step;
                {std::unique_lock lock(mutex);ready.wait(lock,[&]{return stopping||producing>=end||producing-consuming<slots.size();});if(stopping||producing>=end)return;step=producing++;}
                auto batch=make_batch(assets,step,core,count,torch::kCPU);
                {std::lock_guard lock(mutex);if(stopping)return;slots[step%slots.size()]=std::move(batch);}ready.notify_all();
            }}catch(...){std::lock_guard lock(mutex);failure=std::current_exception();stopping=true;ready.notify_all();}
        });}catch(...){finish();throw;}
    }
    void finish(){{std::lock_guard lock(mutex);stopping=true;}ready.notify_all();for(auto& worker:workers)if(worker.joinable())worker.join();}
    ~Prefetch(){finish();}
    Batch next(){std::unique_lock lock(mutex);auto& slot=slots[consuming%slots.size()];ready.wait(lock,[&]{return failure||slot.has_value();});if(failure)std::rethrow_exception(failure);
        auto batch=std::move(*slot);slot.reset();++consuming;lock.unlock();ready.notify_all();return batch;}
};
torch::Tensor forward(Network& net,const Batch& batch){return net->head(net->encode(batch.x,batch.from,batch.to,batch.degree),batch.conditions);}
WeightsData exported(Network& net,const json& provenance) {
    WeightsData w;w.provenance=provenance.dump();torch::NoGradGuard guard;
    for(auto& layer:net->layer)for(auto value:{layer->weight,layer->bias}){auto cpu=value.detach().to(torch::kCPU).contiguous();w.values.insert(w.values.end(),cpu.data_ptr<float>(),cpu.data_ptr<float>()+cpu.numel());}return w;
}
void save_checkpoint(const fs::path& path,Network& net,torch::optim::AdamW& optimizer,uint64_t step) {
    torch::serialize::OutputArchive all,model,state;net->save(model);optimizer.save(state);all.write("model",model);all.write("optimizer",state);all.write("step",torch::tensor(int64_t(step),torch::kInt64));
    auto temp=path;temp+=".part";all.save_to(temp.string());fs::rename(temp,path);
}
uint64_t load_checkpoint(const fs::path& path,Network& net,torch::optim::AdamW& optimizer,torch::Device device) {
    torch::serialize::InputArchive all,model,state;all.load_from(path.string(),device);all.read("model",model);all.read("optimizer",state);net->load(model);optimizer.load(state);torch::Tensor step;all.read("step",step);return step.item<int64_t>();
}
double verify_native(Network& net,const Batch& batch,const WeightsData& w,torch::Device device) {
    torch::NoGradGuard guard;auto& g=batch.first.graph;std::vector<int64_t> from,to;std::vector<float> degree;
    for(uint32_t i=0;i<g.size();++i){degree.push_back(float(std::max(1u,g.offsets[i+1]-g.offsets[i])));for(auto k=g.offsets[i];k<g.offsets[i+1];++k){from.push_back(i);to.push_back(g.neighbors[k]);}}
    auto x=tensor(g.x,{int64_t(g.size()),features},torch::kFloat32,device),src=tensor(from,{int64_t(from.size())},torch::kInt64,device),dst=tensor(to,{int64_t(to.size())},torch::kInt64,device),deg=tensor(degree,{int64_t(g.size()),1},torch::kFloat32,device);
    std::vector<float> cs;for(size_t i=0;i<g.size();++i)cs.insert(cs.end(),batch.first_condition.begin(),batch.first_condition.end());auto cond=tensor(cs,{int64_t(g.size()),neural::conditions},torch::kFloat32,device);
    auto expected=net->head(net->encode(x,src,dst,deg),cond).to(torch::kCPU).contiguous();auto encoded=encode_cuda(g,w,{});auto native=predict_cuda(encoded,batch.first_condition,w,{});
    double maximum=0;auto* reference=expected.data_ptr<float>();for(size_t i=0;i<native.values.size();++i)maximum=std::max(maximum,double(std::abs(native.values[i]-reference[i])));
    if(!std::isfinite(maximum)||maximum>2e-4)throw std::runtime_error("native export differs from LibTorch: "+std::to_string(maximum));return maximum;
}
uint64_t unsigned_option(const char* text) {
    std::string value(text);size_t used=0;
    if(value.empty()||value.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("expected unsigned integer");
    auto result=std::stoull(value,&used);if(used!=value.size())throw std::invalid_argument("invalid integer");return result;
}
int main(int argc,char** argv){try {
    if(argc==2&&std::string_view(argv[1])=="--check-cuda"){
        int devices=0;auto result=cudaGetDeviceCount(&devices);
        if(result==cudaErrorNoDevice||result==cudaErrorInsufficientDriver||(result==cudaSuccess&&!devices))return 77;
        if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
        if(!torch::cuda::is_available())throw NeuralUnavailable("Native CUDA is available but LibTorch CUDA hooks are missing");
        torch::NoGradGuard guard;
        auto values=torch::arange(1,5,torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        if(values.square().sum().item<float>()!=30.f)throw std::runtime_error("LibTorch CUDA arithmetic failed");
        std::cout<<json({{"libtorch_cuda",true},{"devices",devices},{"training_started",false}}).dump()<<std::endl;return 0;
    }
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);if(argc<3)throw std::invalid_argument("blitz-neural-train DATASET RUN [--steps 5120] [--segment-minutes 50] [--core 4096] [--batch 4] [--checkpoint-every 100]");
    fs::path dataset=argv[1],run=argv[2],initialize;uint64_t steps=5120,checkpoint_every=100,core=4096,batch_count=4,workers=2,memory_mib=5120,check_prefetch=0;double minutes=50;
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing option");std::string k=argv[i];if(k=="--steps")steps=unsigned_option(argv[i+1]);else if(k=="--segment-minutes")minutes=std::stod(argv[i+1]);else if(k=="--core")core=unsigned_option(argv[i+1]);else if(k=="--batch")batch_count=unsigned_option(argv[i+1]);else if(k=="--checkpoint-every")checkpoint_every=unsigned_option(argv[i+1]);else if(k=="--workers")workers=unsigned_option(argv[i+1]);else if(k=="--gpu-memory-mib")memory_mib=unsigned_option(argv[i+1]);else if(k=="--check-prefetch")check_prefetch=unsigned_option(argv[i+1]);else if(k=="--initialize")initialize=argv[i+1];else throw std::invalid_argument("unknown option "+k);}
    if(!steps||!checkpoint_every||!core||core>8192||!batch_count||batch_count>128||!workers||workers>4||memory_mib<512||memory_mib>131072||check_prefetch>16||!(minutes>0&&minutes<=50))throw std::invalid_argument("invalid training bounds");
    if(!torch::cuda::is_available())throw NeuralUnavailable("LibTorch CUDA unavailable");torch::set_num_threads(4);torch::manual_seed(0xB1172026);torch::Device device(torch::kCUDA,0);
    if(!check_prefetch){
        size_t free=0,total=0;auto result=cudaMemGetInfo(&free,&total);if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
        if(free<1024ull*1024*1024)throw std::runtime_error("less than 1 GiB GPU memory is available");
        c10::cuda::CUDACachingAllocator::setMemoryFraction(double(std::min<size_t>(memory_mib*1048576,free-512*1024*1024))/total,0);
    }
    auto index=read_json(dataset/"index.json");if(!index.value("complete",false)||index["assets"].empty())throw std::invalid_argument("training dataset is incomplete");
    fs::create_directories(run);json contract={{"dataset_sha256",file_sha256(dataset/"index.json")},{"dataset_contract",index.at("contract_sha256")},{"schema",schema},{"seed",0xB1172026u},{"core",core},{"batch",batch_count},{"workers",workers},{"gpu_memory_mib",memory_mib},{"learning_rate",.001},{"weight_decay",.0001},{"binary_sha256",file_sha256("/proc/self/exe")}};
    if(!initialize.empty())contract["initialize_sha256"]=file_sha256(initialize);
    if(fs::exists(run/"contract.json")&&read_json(run/"contract.json")!=contract)throw std::invalid_argument("training contract differs; use a new run directory");write_json(run/"contract.json",contract);
    std::vector<Asset> assets;for(auto& row:index["assets"]){auto path=dataset/row.at("path").get<std::string>();if(file_sha256(path)!=row.at("sha256").get<std::string>())throw std::invalid_argument("training shard checksum mismatch");assets.push_back(load_asset(path));}
    if(check_prefetch){
        // Nontraining check: compare ordered batches after multiple ring wraparounds.
        Prefetch prefetch(assets,7,7+check_prefetch,uint32_t(core),uint32_t(batch_count),uint32_t(workers));
        for(uint64_t i=7;i<7+check_prefetch;++i){auto actual=prefetch.next(),expected=make_batch(assets,i,uint32_t(core),uint32_t(batch_count),torch::kCPU);
            for(auto member:{&Batch::x,&Batch::from,&Batch::to,&Batch::degree,&Batch::conditions,&Batch::core,&Batch::target})
                if(!torch::equal(actual.*member,expected.*member))throw std::runtime_error("prefetch changed ordered sampling");
            if(actual.first.ids!=expected.first.ids||actual.first_condition!=expected.first_condition)throw std::runtime_error("prefetch changed parity patch");
        }
        std::cout<<json({{"prefetch_checked",check_prefetch},{"workers",workers},{"training_started",false}}).dump()<<std::endl;return 0;
    }
    Network net;net->to(device);
    if(!initialize.empty()){auto w=load_weights(initialize);torch::NoGradGuard guard;size_t at=0;for(auto& layer:net->layer)for(auto value:{layer->weight,layer->bias}){auto t=torch::from_blob(w.values.data()+at,value.sizes(),torch::kFloat32).to(device);value.copy_(t);at+=value.numel();}}
    torch::optim::AdamW optimizer(net->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));uint64_t step=0;
    if(fs::exists(run/"latest.json")){auto last=read_json(run/"latest.json");auto path=run/last.at("checkpoint").get<std::string>();if(file_sha256(path)!=last.at("checkpoint_sha256").get<std::string>())throw std::invalid_argument("checkpoint checksum mismatch");step=load_checkpoint(path,net,optimizer,device);if(step!=last.at("step").get<uint64_t>())throw std::invalid_argument("checkpoint step mismatch");}
    auto start=Clock::now();auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-start).count();};auto start_step=step;double first_loss=0,last_loss=0,loss_sum=0,gradient_norm=0;uint64_t measured=0;
    auto initial=net->parameters().front().detach().clone();std::ofstream log(run/"metrics.jsonl",std::ios::app);Batch batch;
    json health;bool wrote=false;uint64_t trained_vertices=0;double preparation_seconds=0,wait_seconds=0;
    Prefetch prefetch(assets,step,steps,uint32_t(core),uint32_t(batch_count),uint32_t(workers));
    auto checkpoint=[&]{
        auto name="step-"+std::to_string(step);auto weights=exported(net,{{"schema",schema},{"step",step},{"contract",contract},{"training_library",TORCH_VERSION},{"quality","experimental; finite-camera audits required"}});
        double native_error=verify_native(net,batch,weights,device);auto checkpoint_path=run/(name+".pt");save_checkpoint(checkpoint_path,net,optimizer,step);save_weights(run/(name+".blzn"),weights);
        Network restored;restored->to(device);torch::optim::AdamW restored_optimizer(restored->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));if(load_checkpoint(checkpoint_path,restored,restored_optimizer,device)!=step)throw std::runtime_error("restore step mismatch");
        {torch::NoGradGuard guard;for(size_t i=0;i<net->parameters().size();++i){
            auto a=net->parameters()[i],b=restored->parameters()[i];if(!torch::equal(a,b))throw std::runtime_error("checkpoint parameter restore mismatch");
            const auto& before=static_cast<const torch::optim::AdamWParamState&>(*optimizer.state().at(a.unsafeGetTensorImpl()));
            const auto& after=static_cast<const torch::optim::AdamWParamState&>(*restored_optimizer.state().at(b.unsafeGetTensorImpl()));
            if(before.step()!=after.step()||!torch::equal(before.exp_avg(),after.exp_avg())||!torch::equal(before.exp_avg_sq(),after.exp_avg_sq()))throw std::runtime_error("checkpoint optimizer restore mismatch");
        }}
        double changed=(net->parameters().front()-initial).abs().max().item<double>();if(!std::isfinite(changed)||changed<=0)throw std::runtime_error("training did not update parameters");
        auto memory=c10::cuda::CUDACachingAllocator::getDeviceStats(0);double seconds_per_step=elapsed()/std::max<uint64_t>(1,step-start_step);
        health={{"step",step},{"target_steps",steps},{"checkpoint",name+".pt"},{"checkpoint_sha256",file_sha256(checkpoint_path)},{"model",name+".blzn"},{"model_sha256",file_sha256(run/(name+".blzn"))},
            {"finite",std::isfinite(last_loss)&&std::isfinite(gradient_norm)},{"initial_loss",first_loss},{"last_loss",last_loss},{"mean_loss",loss_sum/std::max<uint64_t>(1,measured)},{"gradient_norm",gradient_norm},{"parameter_change",changed},{"native_max_abs_error",native_error},{"restored",true},{"optimizer_restored",true},{"resumed_from_step",start_step},
            {"seconds_per_step",seconds_per_step},{"core_vertices_per_second",trained_vertices/elapsed()},{"mean_preparation_seconds",preparation_seconds/std::max<uint64_t>(1,measured)},{"mean_data_wait_seconds",wait_seconds/std::max<uint64_t>(1,measured)},
            {"remaining_seconds_estimate",seconds_per_step*(steps-step)},{"peak_torch_allocated_mib",memory.allocated_bytes[0].peak/1048576.},{"peak_torch_reserved_mib",memory.reserved_bytes[0].peak/1048576.},{"complete",step>=steps},{"stopped",bool(stopped)}};
        write_json(run/"latest.json",health);std::cout<<health.dump()<<std::endl;wrote=true;
    };
    while(step<steps&&!stopped&&elapsed()<minutes*60) {
        auto begin=Clock::now();batch=prefetch.next();double wait=std::chrono::duration<double>(Clock::now()-begin).count();wait_seconds+=wait;preparation_seconds+=batch.preparation_seconds;
        batch.upload(device);net->train();optimizer.zero_grad();auto predicted=forward(net,batch).index_select(0,batch.core);
        auto label=batch.target.select(1,0);auto positive=label.mean().clamp(.05,.95);auto balance=label*(.5/positive)+(1-label)*(.5/(1-positive));
        auto keep=torch::binary_cross_entropy_with_logits(predicted.select(1,0),label,{}, {},torch::Reduction::None);auto keep_loss=(keep*balance).mean();
        auto displacement=torch::smooth_l1_loss(predicted.slice(1,1,4)*32,batch.target.slice(1,1,4)*32);
        auto loss=keep_loss+.2*displacement;last_loss=loss.item<double>();if(!std::isfinite(last_loss))throw std::runtime_error("nonfinite training loss");if(!measured)first_loss=last_loss;
        loss.backward();gradient_norm=torch::nn::utils::clip_grad_norm_(net->parameters(),1.0);if(!std::isfinite(gradient_norm)||gradient_norm==0)throw std::runtime_error("invalid training gradient");optimizer.step();++step;++measured;loss_sum+=last_loss;
        trained_vertices+=batch.core.numel();double seconds=std::chrono::duration<double>(Clock::now()-begin).count();log<<json({{"step",step},{"loss",last_loss},{"keep_loss",keep_loss.item<double>()},{"displacement_loss",displacement.item<double>()},{"gradient_norm",gradient_norm},{"seconds",seconds},{"preparation_seconds",batch.preparation_seconds},{"data_wait_seconds",wait},{"core_vertices",batch.core.numel()},{"vertices",batch.x.size(0)}}).dump()<<'\n';log.flush();
        if(step%checkpoint_every==0||step==steps)checkpoint();else wrote=false;
        if(step%25==0)std::cout<<json({{"step",step},{"loss",last_loss},{"seconds",seconds}}).dump()<<std::endl;
    }
    if(step>start_step&&!wrote)checkpoint();return step>=steps?0:2;
}catch(const std::exception& e){std::cerr<<"neural training: "<<e.what()<<'\n';return 1;}}
