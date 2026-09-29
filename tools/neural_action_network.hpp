#pragma once
#include "neural_action_data.hpp"
#include "neural_placement.hpp"
#include "neural_network.hpp"
namespace blitz::neural::training {
struct ActionNetworkImpl:torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    uint32_t architecture;
    explicit ActionNetworkImpl(uint32_t version=action_schema):architecture(version){if(!policy_weights(version))throw std::invalid_argument("unsupported action policy architecture");for(unsigned i=0;i<3;++i)layers.push_back(register_module("layer"+std::to_string(i),torch::nn::Linear(i?hidden:policy_inputs(version),i==2?policy_outputs(version):hidden)));}
    torch::Tensor forward(torch::Tensor x){for(unsigned i=0;i<3;++i){x=layers[i]->forward(x);if(i<2)x=torch::relu(x);}return x;}
};TORCH_MODULE(ActionNetwork);
inline WeightsData export_actions(ActionNetwork& model,const json& provenance) {
    WeightsData w;w.architecture=model->architecture;w.provenance=provenance.dump();
    for(auto& p:model->parameters()){auto cpu=p.detach().to(torch::kCPU).contiguous();w.values.insert(w.values.end(),cpu.data_ptr<float>(),cpu.data_ptr<float>()+cpu.numel());}return w;
}
inline torch::Tensor action_loss(const torch::Tensor& prediction,const torch::Tensor& labels,const torch::Tensor& valid,double margin=1,double auxiliary=.25,double penalty=1e-4) {
    auto preferred=labels.bitwise_and(Preferred).ne(0).logical_and(valid),other=preferred.logical_not().logical_and(valid);
    auto pairs=preferred.unsqueeze(2).logical_and(other.unsqueeze(1));auto scores=prediction.select(2,0);
    auto ranking=(torch::relu(margin+scores.unsqueeze(1)-scores.unsqueeze(2))*pairs).sum()/pairs.sum().clamp_min(1);
    auto result=ranking;for(int h=1;h<3;++h){auto target=labels.bitwise_and(h==1?SourcePass:AdjacentPass).ne(0).to(torch::kFloat32);
        result=result+auxiliary*(torch::binary_cross_entropy_with_logits(prediction.select(2,h),target,{}, {},torch::Reduction::None)*valid).sum()/valid.sum().clamp_min(1);}
    return result+penalty*(prediction.square()*valid.unsqueeze(2)).sum()/(valid.sum().clamp_min(1)*action_outputs);
}
inline double action_membership(const torch::Tensor& prediction,const torch::Tensor& labels,const torch::Tensor& valid) {
    auto preferred=labels.bitwise_and(Preferred).ne(0).logical_and(valid),has=preferred.any(1);
    if(!has.any().item<bool>())return 0;
    auto index=prediction.select(2,0).masked_fill(valid.logical_not(),-INFINITY).argmax(1,true);
    return (preferred.gather(1,index).squeeze(1).logical_and(has)).sum().item<double>()/has.sum().item<double>();
}
inline std::vector<double> action_oracle(std::span<const float> x,const WeightsData& w) {
    auto width=policy_inputs(w.architecture),outputs=policy_outputs(w.architecture);
    if(!width||w.values.size()!=policy_weights(w.architecture)||x.size()%width)throw std::invalid_argument("action oracle dimensions");
    std::vector<double> input(x.begin(),x.end());const auto rows=x.size()/width;size_t at=0;
    for(unsigned l=0;l<3;++l){auto in=l?hidden:width,out=l==2?outputs:hidden;std::vector<double> next(rows*out);
        for(size_t row=0;row<rows;++row)for(unsigned j=0;j<out;++j){double sum=w.values[at+size_t(in)*out+j],correction=0;
            for(unsigned k=0;k<in;++k){double term=input[row*in+k]*w.values[at+size_t(j)*in+k]-correction,next_sum=sum+term;correction=(next_sum-sum)-term;sum=next_sum;}
            next[row*out+j]=l<2?std::max(0.,sum):sum;}
        input=std::move(next);at+=size_t(out)*(in+1);}
    return input;
}
}
