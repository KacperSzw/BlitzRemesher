#pragma once
#include "neural_packed.hpp"
#include <cstdint>
#include <bit>
#include <cstddef>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#define BLZ_DATA_HD __host__ __device__
#else
#define BLZ_DATA_HD
#endif
namespace blitz::neural::training {
enum class FeatureEncoding:uint8_t {Half,Snorm,Unorm,Color};
BLZ_DATA_HD constexpr FeatureEncoding feature_encoding(uint32_t c){
    if(c<48||c>=80){auto k=c<48?c%24:(c-80)%24;
        if(k>=6&&k<=8)return FeatureEncoding::Color;
        if((k>=3&&k<=5)||(k>=9&&k<=13))return FeatureEncoding::Snorm;
    }
    if(c==60||c==61||c==62||c==72||c==73)return FeatureEncoding::Snorm;
    if(c==65||c==66)return FeatureEncoding::Unorm;
    return FeatureEncoding::Half;
}
// Slots are split into aligned u16 continuous streams and exact RGB bytes.
BLZ_DATA_HD constexpr int compact_slot(uint32_t channel){
    int s=feature_slot(channel);if(s<0)return s;
    constexpr int first[]={6,23,58,75};int removed=0;
    for(int group=0;group<4;++group){if(s>=first[group]&&s<first[group]+3)return -41-group*3-(s-first[group]);if(s>=first[group]+3)removed+=3;}
    return s-removed;
}
BLZ_DATA_HD constexpr uint32_t compact_width(uint32_t width){return width==80?46:width==128?74:0;}
BLZ_DATA_HD constexpr uint32_t color_width(uint32_t width){return width==80?6:width==128?12:0;}
// IEEE binary16 round-to-nearest-even, shared integer implementation. Infinities
// and values rounded to zero are escaped by the encoder, never saturated.
BLZ_DATA_HD inline uint16_t half_bits(uint32_t bits){
    uint32_t sign=(bits>>16)&0x8000,exponent=(bits>>23)&255,mantissa=bits&0x7fffff;
    if(exponent==255)return uint16_t(sign|0x7c00|(mantissa?0x200:0));
    int e=int(exponent)-127+15;if(e>=31)return uint16_t(sign|0x7c00);
    if(e<=0){if(e< -10)return uint16_t(sign);mantissa|=0x800000;unsigned shift=unsigned(14-e);uint32_t q=mantissa>>shift,tail=mantissa&((1u<<shift)-1),half=1u<<(shift-1);return uint16_t(sign+q+(tail>half||(tail==half&&(q&1))));}
    uint32_t q=mantissa>>13,tail=mantissa&8191;q+=tail>4096||(tail==4096&&(q&1));return uint16_t(sign+(uint32_t(e)<<10)+q);
}
BLZ_DATA_HD inline uint32_t float_bits(uint16_t h){
    uint32_t sign=uint32_t(h&0x8000)<<16,m=h&1023,e=(h>>10)&31;
    if(e==31)return sign|0x7f800000|(m<<13);
    if(e)return sign|((e+112)<<23)|(m<<13);
    if(!m)return sign;int shift=0;while(!(m&1024)){m<<=1;++shift;}return sign|(uint32_t(113-shift)<<23)|((m&1023)<<13);
}
struct CompactView {
    const uint16_t* values{};const uint8_t* colors{};const uint32_t* flags{};const float* conditions{};
    const uint32_t* escape_ids{};const float* escape_values{};uint32_t escapes{},first{};
    const uint32_t* target_offsets{};const int16_t* targets{};
};
BLZ_DATA_HD inline float escape_value(CompactView v,uint32_t key,float ordinary){
    uint32_t lo=0,hi=v.escapes;while(lo<hi){uint32_t mid=lo+(hi-lo)/2;if(v.escape_ids[mid]<key)lo=mid+1;else hi=mid;}
    return lo<v.escapes&&v.escape_ids[lo]==key?v.escape_values[lo]:ordinary;
}
BLZ_DATA_HD inline float half_float(uint16_t h){
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    return std::bit_cast<float>(float_bits(h));
#endif
}
BLZ_DATA_HD inline float compact_feature(CompactView v,uint32_t row,uint32_t channel,uint32_t width){
    int slot=compact_slot(channel);uint32_t r=v.first+row;
    if(slot<=-33&&slot>=-40)return v.conditions[unsigned(-slot-33)];
    if(slot<0&&slot>=-32)return float((v.flags[r]>>unsigned(-slot-1))&1);
    uint32_t key=r*packed_width(width)+uint32_t(feature_slot(channel));
    if(slot<=-41){auto q=v.colors[size_t(r)*color_width(width)+unsigned(-slot-41)];float f=float(q)/255.f;return q==255&&v.escapes?escape_value(v,key,f):f;}
    auto q=v.values[size_t(r)*compact_width(width)+unsigned(slot)];auto type=feature_encoding(channel);float f;bool escaped;
    if(type==FeatureEncoding::Snorm){f=float(int16_t(q))/32767.f;escaped=q==0x8000;}
    else if(type==FeatureEncoding::Unorm){f=float(q)/65535.f;escaped=q==65535;}
    else {f=half_float(q);escaped=q==0x7e00;}
    return escaped&&v.escapes?escape_value(v,key,f):f;
}
BLZ_DATA_HD inline float compact_target(CompactView v,uint32_t row,uint8_t label,uint32_t channel){
    unsigned group=channel/3;if(!(label&(32u<<group)))return 0;unsigned before=0;for(unsigned i=0;i<group;++i)before+=bool(label&(32u<<i));
    auto q=v.targets[v.target_offsets[v.first+row]+before*3+channel%3];return (float(q)/32767.f)*(group?2.f:1.f);
}
}
#undef BLZ_DATA_HD
