#pragma once
#include "neural_update.hpp"
namespace blitz::neural::training {
// Immutable pages own only supervised rows. The stable root is the sole address
// captured by sampler/gather nodes; appending pages never invalidates a graph.
class ResidentDataset {
    struct Page {torch::Tensor values,flags,conditions,labels,placements;};
    std::vector<Page> pages_;
    std::vector<ResidentState> rows_;
    std::vector<std::string> asset_names_,category_names_,asset_categories_;
    std::vector<std::vector<uint32_t>> categories_;
    std::vector<std::array<std::vector<uint32_t>,4>> bins_;
    torch::Tensor states_,categories_device_,assets_device_,bins_device_,ids_device_;
    torch::Device device_;
    uint32_t seed_,architecture_{};
public:
    torch::Tensor root;
    uint32_t maximum_rows{1};
    explicit ResidentDataset(torch::Device device,uint32_t seed):device_(device),seed_(seed){root=torch::empty({int64_t(sizeof(ResidentRoot))},torch::TensorOptions().device(device).dtype(torch::kUInt8));}
    uint32_t states()const{return uint32_t(rows_.size());}
    uint32_t architecture()const{return architecture_;}
    void append(const ActionData& data,const std::string& asset,const std::string& category){
        if(!data.states()||(architecture_&&architecture_!=data.architecture))throw std::invalid_argument("resident dataset architecture or empty page");
        if(uint64_t(states())+data.states()>UINT32_MAX)throw std::length_error("resident state IDs exceed u32");architecture_=data.architecture;uint32_t width=policy_inputs(architecture_),packed=packed_width(width);
        auto asset_it=std::find(asset_names_.begin(),asset_names_.end(),asset);size_t asset_id=size_t(asset_it-asset_names_.begin());
        if(asset_id==asset_names_.size()){
            auto category_it=std::find(category_names_.begin(),category_names_.end(),category);size_t category_id=size_t(category_it-category_names_.begin());if(category_id==category_names_.size()){category_names_.push_back(category);categories_.emplace_back();}
            asset_names_.push_back(asset);asset_categories_.push_back(category);bins_.emplace_back();categories_[category_id].push_back(uint32_t(asset_id));
        }else if(asset_categories_[asset_id]!=category)throw std::invalid_argument("resident asset category changed");
        std::vector<float> values,conditions,placements;std::vector<uint32_t> flags,offsets{0};std::vector<uint8_t> labels;
        for(uint32_t s=0;s<data.states();++s){std::vector<float> x;std::vector<uint8_t> y;
            for(uint32_t i=data.offsets[s];i<data.offsets[s+1];++i){uint8_t label=data.labels[i];if(!(label&(architecture_==placement_schema?248:8)))continue;
                x.insert(x.end(),data.x.begin()+size_t(i)*width,data.x.begin()+size_t(i+1)*width);y.push_back(label);
                if(architecture_==placement_schema)placements.insert(placements.end(),data.targets.begin()+size_t(i)*9,data.targets.begin()+size_t(i+1)*9);}
            if(y.size()>action_pool)throw std::invalid_argument("resident action count exceeds contract");maximum_rows=std::max(maximum_rows,uint32_t(y.size()));
            if(!y.empty()){auto p=pack_actions(x,y,1,uint32_t(y.size()),width);values.insert(values.end(),p.values.begin(),p.values.end());flags.insert(flags.end(),p.flags.begin(),p.flags.end());conditions.insert(conditions.end(),p.conditions.begin(),p.conditions.end());}
            else conditions.resize(conditions.size()+8);
            labels.insert(labels.end(),y.begin(),y.end());offsets.push_back(uint32_t(labels.size()));
        }
        auto upload=[&]<class T>(std::vector<T>& v,torch::ScalarType type){if(v.empty())return torch::empty({0},torch::TensorOptions().device(device_).dtype(type));return torch::from_blob(v.data(),{int64_t(v.size())},type).to(device_);};
        Page page{upload(values,torch::kFloat32),upload(flags,torch::kInt32),upload(conditions,torch::kFloat32),upload(labels,torch::kUInt8),upload(placements,torch::kFloat32)};
        for(uint32_t s=0;s<data.states();++s){auto begin=offsets[s],count=offsets[s+1]-begin;
            rows_.push_back({count?page.values.data_ptr<float>()+size_t(begin)*packed:nullptr,page.conditions.data_ptr<float>()+size_t(s)*8,count&&architecture_==placement_schema?page.placements.data_ptr<float>()+size_t(begin)*9:nullptr,count?reinterpret_cast<uint32_t*>(page.flags.data_ptr<int32_t>())+begin:nullptr,count?page.labels.data_ptr<uint8_t>()+begin:nullptr,count});
            bins_[asset_id][std::min(3u,uint32_t(data.progress[s]*4))].push_back(uint32_t(rows_.size()-1));}
        pages_.push_back(std::move(page));
        // The caller joins the preceding optimizer window before append. Keep
        // old tables alive through the root copy, then reclaim them immediately.
        SamplingTables tables(categories_,bins_,states());auto states=device_records<ResidentState>(rows_,device_),categories=device_records<SampleRange>(tables.categories,device_),assets=device_records<SampleRange>(tables.assets,device_),bins=device_records<SampleRange>(tables.bins,device_),ids=device_records<uint32_t>(tables.states,device_);
        ResidentRoot value{{records<SampleRange>(categories),records<SampleRange>(assets),records<SampleRange>(bins),records<uint32_t>(ids),uint32_t(tables.categories.size()),seed_},records<ResidentState>(states)};
        root.copy_(torch::from_blob(&value,{int64_t(sizeof(value))},torch::kUInt8));states_=std::move(states);categories_device_=std::move(categories);assets_device_=std::move(assets);bins_device_=std::move(bins);ids_device_=std::move(ids);
    }
    size_t bytes()const{size_t result=root.nbytes()+states_.nbytes()+categories_device_.nbytes()+assets_device_.nbytes()+bins_device_.nbytes()+ids_device_.nbytes();for(auto& p:pages_)result+=p.values.nbytes()+p.flags.nbytes()+p.conditions.nbytes()+p.labels.nbytes()+p.placements.nbytes();return result;}
};
class ResidentUpdate {
    ActionNetwork& model_;ResidentDataset& data_;
    uint32_t batch_,pool_{},width_,outputs_;
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
    torch::Tensor losses_,derivative_,placements_;
public:
    DeviceAdam optimizer;torch::Tensor input,target,ids,prediction;
    uint32_t captures{};
    ResidentUpdate(ActionNetwork& model,ResidentDataset& data,uint32_t batch,UpdateSettings settings={}):model_(model),data_(data),batch_(batch),width_(policy_inputs(model->architecture)),outputs_(policy_outputs(model->architecture)),optimizer(model->parameters(),settings){if(!batch||batch>4096||data.architecture()!=model->architecture)throw std::invalid_argument("resident update contract");reserve();}
    ~ResidentUpdate(){cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());graph_.reset();}
    uint32_t pool()const{return pool_;}
    void reserve(){if(pool_>=data_.maximum_rows)return;cuda_check(cudaDeviceSynchronize());graph_.reset();prediction=torch::Tensor{};pool_=data_.maximum_rows;auto options=model_->parameters()[0].options().requires_grad(false);
        input=torch::zeros({batch_,pool_,width_},options);target=torch::zeros({batch_,pool_},options.dtype(torch::kUInt8));ids=torch::zeros({batch_},options.dtype(torch::kInt32));losses_=torch::empty({batch_},options);derivative_=torch::zeros({batch_,pool_,outputs_},options);if(model_->architecture==placement_schema)placements_=torch::empty({batch_,pool_,9},options);
    }
    void gather(){sample_resident_update(records<ResidentRoot>(data_.root),input.data_ptr<float>(),target.data_ptr<uint8_t>(),placements_.defined()?placements_.data_ptr<float>():nullptr,reinterpret_cast<uint32_t*>(ids.data_ptr<int32_t>()),records<UpdateState>(optimizer.control),batch_,pool_,width_,c10::cuda::getCurrentCUDAStream());cuda_check(cudaGetLastError());}
    void eager(){auto stream=c10::cuda::getCurrentCUDAStream();gather();optimizer.zero_grad();prediction=model_->forward(input);
        if(placements_.defined())placement_loss_update(prediction.data_ptr<float>(),target.data_ptr<uint8_t>(),placements_.data_ptr<float>(),nullptr,derivative_.data_ptr<float>(),losses_.data_ptr<float>(),records<UpdateState>(optimizer.control),batch_,pool_,optimizer.settings,stream);
        else loss_update(prediction.data_ptr<float>(),target.data_ptr<uint8_t>(),derivative_.data_ptr<float>(),losses_.data_ptr<float>(),records<UpdateState>(optimizer.control),batch_,pool_,optimizer.settings,stream);
        prediction.backward(derivative_);optimizer.update();}
    void capture(){reserve();if(graph_)return;auto saved=optimizer.snapshot();cuda_check(cudaDeviceSynchronize());{c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());for(unsigned i=0;i<3;++i)eager();cuda_check(cudaStreamSynchronize(stream.current_stream()));optimizer.restore(saved);graph_=std::make_unique<at::cuda::CUDAGraph>();graph_->capture_begin();eager();graph_->capture_end();optimizer.restore(saved);cuda_check(cudaStreamSynchronize(stream.current_stream()));}++captures;}
    void run(uint32_t count){capture();for(uint32_t i=0;i<count;++i)graph_->replay();}
};
}
