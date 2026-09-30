#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace blitz::neural::training {
// Persisted absolute deadlines include downtime between resumptions. The caller
// supplies time so boundaries are testable without a long-running experiment.
struct CycleTime {
    int64_t start_ms{},work_end_ms{},end_ms{};
    static CycleTime duration(int64_t start,uint32_t seconds,uint32_t reserve){
        if(start<0||!seconds||reserve>=seconds||start>INT64_MAX-int64_t(seconds)*1000)throw std::invalid_argument("cycle duration contract");
        return {start,start+int64_t(seconds-reserve)*1000,start+int64_t(seconds)*1000};
    }
    bool finishing(int64_t now)const{return end_ms&&now>=work_end_ms;}
    bool expired(int64_t now)const{return end_ms&&now>=end_ms;}
};
inline uint32_t update_target(uint32_t step,uint32_t count){if(count>UINT32_MAX-step)throw std::overflow_error("optimizer u32 step counter exhausted");return step+count;}
}
