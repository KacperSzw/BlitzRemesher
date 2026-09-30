#pragma once
#include "neural_update.hpp"
namespace blitz::neural::training {
// Immutable pages own only supervised rows. The stable root is the sole address
// captured by sampler/gather nodes; appending pages never invalidates a graph.
class ResidentDataset {
    struct Page {torch::Tensor values,flags,conditions,labels,placements,colors,escape_ids,escape_values,target_offsets;};
    std::vector<ResidentPage> page_records_;
    std::vector<Page> pages_;
    std::vector<uint32_t> page_states_;std::vector<std::array<torch::Tensor,4>> bin_ids_;
    std::vector<std::array<bool,4>> dirty_bins_;size_t published_rows_{};uint64_t evicted_{};
    std::vector<ResidentState> rows_;
    std::vector<std::string> asset_names_,category_names_,asset_categories_;
    std::vector<std::vector<uint32_t>> categories_;
    std::vector<std::array<std::vector<uint32_t>,4>> bins_;
    torch::Tensor pages_device_,states_,categories_device_,assets_device_,bins_device_,ids_device_,bin_pointers_;
    torch::Device device_;
    uint32_t seed_,architecture_{};bool compact_;size_t limit_;
public:
    torch::Tensor root;
    uint32_t maximum_rows{1};
    explicit ResidentDataset(torch::Device device,uint32_t seed,bool compact=true,size_t limit=SIZE_MAX):device_(device),seed_(seed),compact_(compact),limit_(limit){root=torch::empty({int64_t(sizeof(ResidentRoot))},torch::TensorOptions().device(device).dtype(torch::kUInt8));}
    uint32_t states()const{return uint32_t(rows_.size());}
    uint32_t architecture()const{return architecture_;}
    uint64_t evicted()const{return evicted_;}
    size_t pages()const{return pages_.size();}
    json order()const{json value=json::array();for(size_t i=0;i<asset_names_.size();++i)value.push_back({asset_names_[i],asset_categories_[i]});return value;}
    void restore_order(const json& value){if(states())throw std::logic_error("sampler order must precede replay pages");for(const auto& pair:value)asset_index(pair.at(0).get<std::string>(),pair.at(1).get<std::string>());}
    void clear(){cuda_check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));rows_.clear();page_records_.clear();pages_.clear();page_states_.clear();bin_ids_.clear();dirty_bins_.clear();published_rows_=0;asset_names_.clear();category_names_.clear();asset_categories_.clear();categories_.clear();bins_.clear();maximum_rows=1;states_=pages_device_=categories_device_=assets_device_=bins_device_=ids_device_=bin_pointers_=torch::Tensor{};}
    void append(const ActionData& data,const std::string& asset,const std::string& category){
        if(compact_){append(compact_actions(data),asset,category);return;}
        validate_actions(data);
        const size_t required=data.labels.size()*(packed_width(policy_inputs(data.architecture))*4+5+(data.architecture==placement_schema?36:0))+data.states()*(48+sizeof(ResidentState)*3)+sizeof(ResidentPage)*2+4096;
        make_room(required);
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
        pages_.push_back(std::move(page));page_states_.push_back(data.states());dirty_bins_[asset_id].fill(true);
        publish();
    }
    // Only the learner mutates these tables, between update windows on its own
    // stream. Changed bin buffers and append-only state tails are incremental.
    template<class T> void table(torch::Tensor& destination,std::span<const T> values,size_t first=0){
        auto required=values.size_bytes();if(!destination.defined()||size_t(destination.numel())<required){
            size_t capacity=(required+255)&~size_t(255),old=destination.defined()?size_t(destination.numel()):0;
            auto growing=std::max(capacity,((old+old/2)+255)&~size_t(255)),used=bytes();
            if(growing-capacity<=limit_-std::min(limit_,used+4096))capacity=growing;
            destination=torch::empty({int64_t(capacity)},root.options());first=0;}
        if(first<values.size()){auto host=torch::from_blob(const_cast<T*>(values.data()+first),{int64_t((values.size()-first)*sizeof(T))},torch::kUInt8);destination.slice(0,first*sizeof(T),required).copy_(host);}
    }
    void publish(){
        std::vector<SampleRange> categories,assets,bins;std::vector<const uint32_t*> pointers;
        for(auto& group:categories_){SampleRange category{uint32_t(assets.size()),0};for(auto asset:group){SampleRange a{uint32_t(bins.size()),0};
            for(size_t b=0;b<4;++b)if(!bins_[asset][b].empty()){
                if(dirty_bins_[asset][b]){table<uint32_t>(bin_ids_[asset][b],bins_[asset][b]);dirty_bins_[asset][b]=false;}
                bins.push_back({0,uint32_t(bins_[asset][b].size())});pointers.push_back(records<uint32_t>(bin_ids_[asset][b]));++a.count;}
            if(a.count){assets.push_back(a);++category.count;}}
            if(category.count)categories.push_back(category);}
        table<ResidentState>(states_,rows_,published_rows_);published_rows_=rows_.size();table<ResidentPage>(pages_device_,page_records_);
        table<SampleRange>(categories_device_,categories);table<SampleRange>(assets_device_,assets);table<SampleRange>(bins_device_,bins);table<const uint32_t*>(bin_pointers_,pointers);
        ResidentRoot value{{records<SampleRange>(categories_device_),records<SampleRange>(assets_device_),records<SampleRange>(bins_device_),nullptr,uint32_t(categories.size()),seed_},records<ResidentState>(states_),records<ResidentPage>(pages_device_),records<const uint32_t*>(bin_pointers_)};
        root.copy_(torch::from_blob(&value,{int64_t(sizeof(value))},torch::kUInt8));
    }
    void make_room(size_t required){
        if(required>limit_)throw std::length_error("one replay page exceeds dataset budget");
        while(required>limit_-std::min(limit_,bytes())&&!pages_.empty()){
            cuda_check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));auto count=page_states_.front();
            pages_.erase(pages_.begin());page_records_.erase(page_records_.begin());page_states_.erase(page_states_.begin());rows_.erase(rows_.begin(),rows_.begin()+count);for(auto& row:rows_)--row.page;
            for(size_t a=0;a<bins_.size();++a)for(size_t b=0;b<4;++b){auto& bin=bins_[a][b];auto at=std::lower_bound(bin.begin(),bin.end(),count);bin.erase(bin.begin(),at);for(auto& id:bin)id-=count;dirty_bins_[a][b]=true;bin_ids_[a][b]=torch::Tensor{};}
            published_rows_=0;++evicted_;states_=pages_device_=categories_device_=assets_device_=bins_device_=ids_device_=bin_pointers_=torch::Tensor{};
        }
        if(required>limit_-std::min(limit_,bytes()))throw std::length_error("resident replay table budget exhausted");
    }
    size_t asset_index(const std::string& asset,const std::string& category){
        auto asset_it=std::find(asset_names_.begin(),asset_names_.end(),asset);size_t asset_id=size_t(asset_it-asset_names_.begin());
        if(asset_id==asset_names_.size()){
            auto category_it=std::find(category_names_.begin(),category_names_.end(),category);size_t category_id=size_t(category_it-category_names_.begin());if(category_id==category_names_.size()){category_names_.push_back(category);categories_.emplace_back();}
            asset_names_.push_back(asset);asset_categories_.push_back(category);bins_.emplace_back();bin_ids_.emplace_back();dirty_bins_.emplace_back();categories_[category_id].push_back(uint32_t(asset_id));
        }else if(asset_categories_[asset_id]!=category)throw std::invalid_argument("resident asset category changed");
        return asset_id;
    }
    void append(const CompactActions& supplied,const std::string& asset,const std::string& category){
        validate_compact(supplied);if(!compact_){append(expand_compact(supplied),asset,category);return;}
        std::optional<CompactActions> filtered;if(std::any_of(supplied.labels.begin(),supplied.labels.end(),[&](uint8_t x){return !(x&(supplied.architecture==placement_schema?248:8));}))filtered=supervised_compact(supplied);const auto& data=filtered?*filtered:supplied;
        if(!data.states()||(architecture_&&architecture_!=data.architecture)||uint64_t(states())+data.states()>UINT32_MAX)throw std::invalid_argument("compact resident architecture/state domain");
        make_room(data.bytes()+data.states()*(sizeof(ResidentState)*3+16)+sizeof(ResidentPage)*2+4096);
        architecture_=data.architecture;auto asset_id=asset_index(asset,category);
        auto upload=[&]<class T>(const std::vector<T>& v,torch::ScalarType type){if(v.empty())return torch::empty({0},torch::TensorOptions().device(device_).dtype(type));return torch::from_blob(const_cast<T*>(v.data()),{int64_t(v.size())},type).to(device_);};
        Page p{upload(data.values,torch::kInt16),upload(data.flags,torch::kInt32),upload(data.conditions,torch::kFloat32),upload(data.labels,torch::kUInt8),upload(data.targets,torch::kInt16),upload(data.colors,torch::kUInt8),upload(data.escape_ids,torch::kInt32),upload(data.escape_values,torch::kFloat32),upload(data.target_offsets,torch::kInt32)};
        CompactView v{reinterpret_cast<uint16_t*>(p.values.data_ptr<int16_t>()),p.colors.data_ptr<uint8_t>(),reinterpret_cast<uint32_t*>(p.flags.data_ptr<int32_t>()),p.conditions.data_ptr<float>(),reinterpret_cast<uint32_t*>(p.escape_ids.data_ptr<int32_t>()),p.escape_values.data_ptr<float>(),uint32_t(data.escape_ids.size()),0,reinterpret_cast<uint32_t*>(p.target_offsets.data_ptr<int32_t>()),p.placements.data_ptr<int16_t>()};
        page_records_.push_back({nullptr,v.conditions,nullptr,v.flags,p.labels.data_ptr<uint8_t>(),v,true});
        for(uint32_t s=0;s<data.states();++s){auto first=data.offsets[s],count=data.offsets[s+1]-first;maximum_rows=std::max(maximum_rows,count);rows_.push_back({uint32_t(pages_.size()),first,s,count});bins_[asset_id][std::min(3u,uint32_t(data.progress[s]*4))].push_back(uint32_t(rows_.size()-1));}
        pages_.push_back(std::move(p));page_states_.push_back(data.states());dirty_bins_[asset_id].fill(true);publish();
    }
    size_t bytes()const{auto n=[](const torch::Tensor& x){return x.defined()?x.nbytes():0;};size_t result=root.nbytes()+n(pages_device_)+n(states_)+n(categories_device_)+n(assets_device_)+n(bins_device_)+n(ids_device_)+n(bin_pointers_);for(auto& p:pages_)result+=p.values.nbytes()+p.flags.nbytes()+p.conditions.nbytes()+p.labels.nbytes()+p.placements.nbytes()+(p.colors.defined()?p.colors.nbytes()+p.escape_ids.nbytes()+p.escape_values.nbytes()+p.target_offsets.nbytes():0);for(auto& a:bin_ids_)for(auto& b:a)result+=n(b);return result;}
};
class ResidentUpdate {
    ActionNetwork& model_;ResidentDataset& data_;
    uint32_t batch_,pool_{},width_,outputs_;
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
    torch::Tensor losses_,derivative_,placements_;
    UpdateBackend backend_;
    torch::Tensor hidden_[2],delta_[2],workspace_;
    std::unique_ptr<FusedMlp,decltype(&destroy_fused_mlp)> fused_{nullptr,destroy_fused_mlp};
public:
    DeviceAdam optimizer;torch::Tensor input,target,ids,prediction;
    uint32_t captures{};
    ResidentUpdate(ActionNetwork& model,ResidentDataset& data,uint32_t batch,UpdateSettings settings={},UpdateBackend backend=UpdateBackend::Reference):model_(model),data_(data),batch_(batch),width_(policy_inputs(model->architecture)),outputs_(policy_outputs(model->architecture)),backend_(backend),optimizer(model->parameters(),settings){if(!batch||batch>4096||data.architecture()!=model->architecture||(backend==UpdateBackend::Fused&&model->architecture!=placement_schema))throw std::invalid_argument("resident update contract");reserve();}
    ~ResidentUpdate(){cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());graph_.reset();}
    uint32_t pool()const{return pool_;}
    uint32_t fused_epilogues()const{return fused_?fused_mlp_epilogues(fused_.get()):0;}
    std::array<int32_t,3> fused_algorithms()const{return fused_?fused_mlp_algorithms(fused_.get()):std::array<int32_t,3>{-1,-1,-1};}
    void reserve(){if(pool_>=data_.maximum_rows)return;cuda_check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));graph_.reset();fused_.reset();prediction=torch::Tensor{};pool_=data_.maximum_rows;auto options=model_->parameters()[0].options().requires_grad(false);
        input=torch::zeros({batch_,pool_,width_},options);target=torch::zeros({batch_,pool_},options.dtype(torch::kUInt8));ids=torch::zeros({batch_},options.dtype(torch::kInt32));losses_=torch::empty({batch_},options);derivative_=torch::zeros({batch_,pool_,outputs_},options);if(model_->architecture==placement_schema)placements_=torch::empty({batch_,pool_,9},options);
        if(backend_==UpdateBackend::Fused){prediction=torch::empty({batch_,pool_,outputs_},options);FusedMlpBuffers b;b.width=model_->hidden_width;b.rows=batch_*pool_;b.input=input.data_ptr<float>();b.output=prediction.data_ptr<float>();b.derivative=derivative_.data_ptr<float>();
            for(unsigned i=0;i<2;++i){hidden_[i]=torch::empty({b.rows,b.width},options);delta_[i]=torch::empty_like(hidden_[i]);b.hidden[i]=hidden_[i].data_ptr<float>();b.delta[i]=delta_[i].data_ptr<float>();}
            workspace_=torch::empty({4<<20},options.dtype(torch::kUInt8));b.workspace=workspace_.data_ptr();b.workspace_bytes=workspace_.nbytes();
            for(unsigned i=0;i<6;++i){b.parameter[i]=optimizer.parameters[i].data_ptr<float>();b.gradient[i]=optimizer.parameters[i].grad().data_ptr<float>();}fused_.reset(create_fused_mlp(b));}
    }
    void gather(){sample_resident_update(records<ResidentRoot>(data_.root),input.data_ptr<float>(),target.data_ptr<uint8_t>(),placements_.defined()?placements_.data_ptr<float>():nullptr,reinterpret_cast<uint32_t*>(ids.data_ptr<int32_t>()),records<UpdateState>(optimizer.control),batch_,pool_,width_,c10::cuda::getCurrentCUDAStream());cuda_check(cudaGetLastError());}
    void eager(){auto stream=c10::cuda::getCurrentCUDAStream();gather();if(fused_)fused_mlp_forward(fused_.get(),stream);else {optimizer.zero_grad();prediction=model_->forward(input);}
        if(placements_.defined())placement_loss_update(prediction.data_ptr<float>(),target.data_ptr<uint8_t>(),placements_.data_ptr<float>(),nullptr,derivative_.data_ptr<float>(),losses_.data_ptr<float>(),records<UpdateState>(optimizer.control),batch_,pool_,optimizer.settings,stream);
        else loss_update(prediction.data_ptr<float>(),target.data_ptr<uint8_t>(),derivative_.data_ptr<float>(),losses_.data_ptr<float>(),records<UpdateState>(optimizer.control),batch_,pool_,optimizer.settings,stream);
        if(fused_)fused_mlp_backward(fused_.get(),stream);else prediction.backward(derivative_);optimizer.update();}
    void capture(){reserve();if(graph_)return;auto saved=optimizer.snapshot();cuda_check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));{std::optional<c10::cuda::CUDAStreamGuard> guard;if(c10::cuda::getCurrentCUDAStream()==c10::cuda::getDefaultCUDAStream())guard.emplace(c10::cuda::getStreamFromPool());auto stream=c10::cuda::getCurrentCUDAStream();for(unsigned i=0;i<3;++i)eager();cuda_check(cudaStreamSynchronize(stream));optimizer.restore(saved);graph_=std::make_unique<at::cuda::CUDAGraph>();graph_->capture_begin({},cudaStreamCaptureModeThreadLocal);eager();graph_->capture_end();optimizer.restore(saved);cuda_check(cudaStreamSynchronize(stream));}++captures;}
    void run(uint32_t count){if(!data_.states())throw std::invalid_argument("resident update requires a populated dataset window");capture();for(uint32_t i=0;i<count;++i)graph_->replay();}
};
}
