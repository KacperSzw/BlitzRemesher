#pragma once
#include "neural_data.hpp"
#include "neural_placement.hpp"
namespace blitz::neural::training {
constexpr uint8_t SourcePass=1,AdjacentPass=2,Preferred=4,Queried=8;
constexpr uint32_t action_pool=16;
struct ActionData {
    uint32_t architecture{action_schema};
    std::vector<float> x,progress,targets;
    std::vector<uint8_t> labels;
    std::vector<uint32_t> offsets{0},from,to;
    size_t states() const {return offsets.size()-1;}
};
inline void validate_actions(const ActionData& a) {
    auto width=policy_inputs(a.architecture);if(!width||a.offsets.empty()||a.offsets.front()!=0||a.offsets.back()!=a.labels.size()||!std::is_sorted(a.offsets.begin(),a.offsets.end())||a.x.size()!=a.labels.size()*width||a.from.size()!=a.labels.size()||a.to.size()!=a.labels.size()||a.progress.size()!=a.states()||a.targets.size()!=(a.architecture==placement_schema?a.labels.size()*9:0))throw std::invalid_argument("action shard dimensions");
    for(size_t i=0;i<a.states();++i)if(a.offsets[i+1]-a.offsets[i]<1||a.offsets[i+1]-a.offsets[i]>action_pool||!std::isfinite(a.progress[i])||a.progress[i]<0||a.progress[i]>1)throw std::invalid_argument("action state dimensions/progress");
    for(float x:a.x)if(!std::isfinite(x))throw std::invalid_argument("nonfinite action feature");
    for(float x:a.targets)if(!std::isfinite(x))throw std::invalid_argument("nonfinite placement target");
    for(uint8_t x:a.labels){if(a.architecture==action_schema){if(x>15||!(x&Queried)||((x&Preferred)&&((x&3)!=3)))throw std::invalid_argument("invalid action supervision mask");}
        else if(((x&1)&&!(x&SourceKnown))||((x&2)&&!(x&AdjacentKnown))||((x&(Preferred|PositionKnown))&&((x&27)!=27))||((x&(Normal0Known|Normal1Known))&&!(x&PositionKnown)))throw std::invalid_argument("invalid placement supervision mask");}
    if(a.architecture==placement_schema)for(size_t i=0;i<a.labels.size();++i)for(unsigned j=0;j<9;++j){float x=a.targets[i*9+j];bool known=a.labels[i]&(32u<<(j/3));if((!known&&x!=0)||(known&&std::abs(x)>(j<3?1.f:2.f)))throw std::invalid_argument("placement target outside contract");}
}
}
