#pragma once
#include "neural_packed.hpp"
#include <bit>
#include <span>
#include <vector>
#include <stdexcept>
#include <cmath>
namespace blitz::neural::training {
struct PackedActions {std::vector<float> values,conditions;std::vector<uint32_t> flags;};
inline PackedActions pack_actions(std::span<const float> x,std::span<const uint8_t> labels,uint32_t states,uint32_t pool,uint32_t width){
    auto packed=packed_width(width);if(!packed||!pool||x.size()!=size_t(states)*pool*width||labels.size()!=size_t(states)*pool)throw std::invalid_argument("packed action dimensions");PackedActions out;out.values.resize(size_t(states)*pool*packed);out.conditions.resize(size_t(states)*8);out.flags.resize(size_t(states)*pool);
    for(uint32_t state=0;state<states;++state){bool condition_set=false;for(uint32_t row=0;row<pool;++row){size_t id=size_t(state)*pool+row;if(!labels[id])continue;const float* values=x.data()+id*width;
        for(uint32_t c=0;c<width;++c){float value=values[c];if(!std::isfinite(value))throw std::invalid_argument("nonfinite packed feature");auto slot=feature_slot(c);
            if(slot>=0){if(c==78||c==79){if(std::bit_cast<uint32_t>(value)!=std::bit_cast<uint32_t>(values[c==78?23:47]))throw std::invalid_argument("duplicate feature differs");}out.values[id*packed+unsigned(slot)]=value;}
            else if(slot>=-32){if(value!=0&&value!=1)throw std::invalid_argument("feature flag outside 0..1");if(value==1)out.flags[id]|=1u<<unsigned(-slot-1);}
            else {auto& shared=out.conditions[size_t(state)*8+unsigned(-slot-33)];if(condition_set&&std::bit_cast<uint32_t>(shared)!=std::bit_cast<uint32_t>(value))throw std::invalid_argument("state conditions differ between actions");shared=value;}}
        condition_set=true;}}
    return out;
}
}
