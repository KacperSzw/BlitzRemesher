#pragma once
#include "neural_update.hpp"
namespace blitz::neural::training {
// Immutable pages own only supervised rows. The stable root is the sole address
// captured by sampler/gather nodes; appending pages never invalidates a graph.
class ResidentDataset {
    struct Page {torch::Tensor values,flags,conditions,labels,placements,colors,escape_ids,escape_values,target_offsets;};
    std::vector<ResidentPage> page_records_;
    std::vector<Page> pages_;
    std::vector<ResidentState> rows_;
    std::vector<std::string> asset_names_,category_names_,asset_categories_;
    std::vector<std::vector<uint32_t>> categories_;
    std::vector<std::array<std::vector<uint32_t>,4>> bins_;
    torch::Tensor pages_device_,states_,categories_device_,assets_device_,bins_device_,ids_device_;
    torch::Device device_;
    uint32_t seed_,architecture_{};bool compact_;size_t limit_;
public:
    torch::Tensor root;
    uint32_t maximum_rows{1};
    explicit ResidentDataset(torch::Device device,uint32_t seed,bool compact=true,size_t limit=SIZE_MAX):device_(device),seed_(seed),compact_(compact),limit_(limit){root=torch::empty({int64_t(sizeof(ResidentRoot))},torch::TensorOptions().device(device).dtype(torch::kUInt8));}
    uint32_t states()const{return uint32_t(rows_.size());}
    uint32_t architecture()const{return architecture_;}
    void clear(){cuda_check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));rows_.clear();page_records_.clear();pages_.clear();asset_names_.clear();category_names_.clear();asset_categories_.clear();categories_.clear();bins_.clear();maximum_rows=1;states_=pages_device_=categories_device_=assets_device_=bins_device_=ids_device_=torch::Tensor{};}
    void append(const ActionData& data,const std::string& asset,const std::string& category){
        if(compact_){append(compact_actions(data),asset,category);return;}
        validate_actions(data);
        const size_t required=data.labels.size()*(packed_width(policy_inputs(data.architecture))*4+5+(data.architecture==placement_schema?36:0))+data.states()*(36+sizeof(ResidentState))+sizeof(ResidentPage)+48;
        if(required>limit_-std::min(limit_,bytes()))throw std::length_error("resident dataset memory budget exhausted; start a bounded shard window");
        if(!data.states()||(architecture_&&architecture_!=data.architecture))throw std::invalid_argument("resident dataset architecture or empty page");
        if(uint64_t(states())+data.states()>UINT32_MAX)throw std::length_error("resident state IDs exceed u32");architecture_=data.architecture;uint32_t width=policy_inputs(architecture_),packed=packed_width(width);
        size_t asset_id=asset_index(asset,category);
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
            rows_.push_back({uint32_t(pages_.size()),begin,s,count});
            bins_[asset_id][std::min(3u,uint32_t(data.progress[s]*4))].push_back(uint32_t(rows_.size()-1));}
        page_records_.push_back({page.values.data_ptr<float>(),page.conditions.data_ptr<float>(),page.placements.data_ptr<float>(),reinterpret_cast<uint32_t*>(page.flags.data_ptr<int32_t>()),page.labels.data_ptr<uint8_t>(),{},false});
        pages_.push_back(std::move(page));
        publish();
    }
    void publish(){
        // The caller joins the preceding optimizer window before append. Keep
        // old tables alive through the root copy, then reclaim them immediately.
        SamplingTables tables(categories_,bins_,states());auto states=device_records<ResidentState>(rows_,device_),categories=device_records<SampleRange>(tables.categories,device_),assets=device_records<SampleRange>(tables.assets,device_),bins=device_records<SampleRange>(tables.bins,device_),ids=device_records<uint32_t>(tables.states,device_);
        auto pages=device_records<ResidentPage>(page_records_,device_);
        ResidentRoot value{{records<SampleRange>(categories),records<SampleRange>(assets),records<SampleRange>(bins),records<uint32_t>(ids),uint32_t(tables.categories.size()),seed_},records<ResidentState>(states),records<ResidentPage>(pages)};
        root.copy_(torch::from_blob(&value,{int64_t(sizeof(value))},torch::kUInt8));pages_device_=std::move(pages);states_=std::move(states);categories_device_=std::move(categories);assets_device_=std::move(assets);bins_device_=std::move(bins);ids_device_=std::move(ids);
    }
    size_t asset_index(const std::string& asset,const std::string& category){
        auto asset_it=std::find(asset_names_.begin(),asset_names_.end(),asset);size_t asset_id=size_t(asset_it-asset_names_.begin());
        if(asset_id==asset_names_.size()){
            auto category_it=std::find(category_names_.begin(),category_names_.end(),category);size_t category_id=size_t(category_it-category_names_.begin());if(category_id==category_names_.size()){category_names_.push_back(category);categories_.emplace_back();}
            asset_names_.push_back(asset);asset_categories_.push_back(category);bins_.emplace_back();categories_[category_id].push_back(uint32_t(asset_id));
        }else if(asset_categories_[asset_id]!=category)throw std::invalid_argument("resident asset category changed");
        return asset_id;
    }
    void append(const CompactActions& supplied,const std::string& asset,const std::string& category){
        validate_compact(supplied);if(!compact_){append(expand_compact(supplied),asset,category);return;}
        std::optional<CompactActions> filtered;if(std::any_of(supplied.labels.begin(),supplied.labels.end(),[&](uint8_t x){return !(x&(supplied.architecture==placement_schema?248:8));}))filtered=supervised_compact(supplied);const auto& data=filtered?*filtered:supplied;
        if(!data.states()||(architecture_&&architecture_!=data.architecture)||uint64_t(states())+data.states()>UINT32_MAX)throw std::invalid_argument("compact resident architecture/state domain");
        if(data.bytes()+data.states()*sizeof(ResidentState)+sizeof(ResidentPage)+48>limit_-std::min(limit_,bytes()))throw std::length_error("resident dataset memory budget exhausted; start a bounded shard window");
        architecture_=data.architecture;auto asset_id=asset_index(asset,category);
        auto upload=[&]<class T>(const std::vector<T>& v,torch::ScalarType type){if(v.empty())return torch::empty({0},torch::TensorOptions().device(device_).dtype(type));return torch::from_blob(const_cast<T*>(v.data()),{int64_t(v.size())},type).to(device_);};
        Page p{upload(data.values,torch::kInt16),upload(data.flags,torch::kInt32),upload(data.conditions,torch::kFloat32),upload(data.labels,torch::kUInt8),upload(data.targets,torch::kInt16),upload(data.colors,torch::kUInt8),upload(data.escape_ids,torch::kInt32),upload(data.escape_values,torch::kFloat32),upload(data.target_offsets,torch::kInt32)};
        CompactView v{reinterpret_cast<uint16_t*>(p.values.data_ptr<int16_t>()),p.colors.data_ptr<uint8_t>(),reinterpret_cast<uint32_t*>(p.flags.data_ptr<int32_t>()),p.conditions.data_ptr<float>(),reinterpret_cast<uint32_t*>(p.escape_ids.data_ptr<int32_t>()),p.escape_values.data_ptr<float>(),uint32_t(data.escape_ids.size()),0,reinterpret_cast<uint32_t*>(p.target_offsets.data_ptr<int32_t>()),p.placements.data_ptr<int16_t>()};
        page_records_.push_back({nullptr,v.conditions,nullptr,v.flags,p.labels.data_ptr<uint8_t>(),v,true});
        for(uint32_t s=0;s<data.states();++s){auto first=data.offsets[s],count=data.offsets[s+1]-first;maximum_rows=std::max(maximum_rows,count);rows_.push_back({uint32_t(pages_.size()),first,s,count});bins_[asset_id][std::min(3u,uint32_t(data.progress[s]*4))].push_back(uint32_t(rows_.size()-1));}
        pages_.push_back(std::move(p));publish();
    }
    size_t bytes()const{auto n=[](const torch::Tensor& x){return x.defined()?x.nbytes():0;};size_t result=root.nbytes()+n(pages_device_)+n(states_)+n(categories_device_)+n(assets_device_)+n(bins_device_)+n(ids_device_);for(auto& p:pages_)result+=p.values.nbytes()+p.flags.nbytes()+p.conditions.nbytes()+p.labels.nbytes()+p.placements.nbytes()+(p.colors.defined()?p.colors.nbytes()+p.escape_ids.nbytes()+p.escape_values.nbytes()+p.target_offsets.nbytes():0);return result;}
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
    void run(uint32_t count){if(!data_.states())throw std::invalid_argument("resident update requires a populated dataset window");capture();for(uint32_t i=0;i<count;++i)graph_->replay();}
};
}
