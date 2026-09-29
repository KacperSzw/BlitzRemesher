#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define BLITZ_PACKED_HD __host__ __device__
#else
#define BLITZ_PACKED_HD
#endif
namespace blitz::neural::training {
// Nonnegative: continuous FP32 slot; -1..-32: packed flag; -33..-40:
// per-state condition. Duplicate mean-edge channels refer to the original slot.
BLITZ_PACKED_HD constexpr int feature_slot(uint32_t channel){
    if(channel>=80){auto block=(channel-80)/24,local=(channel-80)%24;if(local>=15&&local<=21)return -1-int(18+block*7+local-15);return int(52+block*17+local-(local>21?7:0));}
    if(channel<15)return int(channel);if(channel<22)return -1-int(channel-15);
    if(channel<39)return int(channel-7);if(channel<46)return -1-int(7+channel-39);
    if(channel<48)return int(channel-14);if(channel<56)return -33-int(channel-48);
    if(channel<67)return int(channel-22);if(channel<69)return -15-int(channel-67);
    if(channel<75)return int(channel-24);if(channel<77)return -17-int(channel-75);
    if(channel==77)return 51;return channel==78?16:33;
}
BLITZ_PACKED_HD constexpr uint32_t packed_width(uint32_t width){return width==80?52:width==128?86:0;}
}
#undef BLITZ_PACKED_HD
