#pragma once
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace blitz::neural::training {
struct CheckpointCadence {
    int64_t period_ms,next_ms;
    CheckpointCadence(int64_t now,uint32_t seconds):period_ms(int64_t(seconds)*1000),next_ms(now+period_ms){if(!seconds||seconds>3600)throw std::invalid_argument("checkpoint interval outside 1..3600 seconds");}
    bool due(int64_t now)const{return now>=next_ms;}
    void submitted(int64_t now){next_ms=now+period_ms;}
};
// Learning and finalization are separate budgets. Recovery resumes from durable
// completed learning time; downtime does not count as training.
struct CycleTime {
    int64_t start_ms{},work_end_ms{},end_ms{};
    static CycleTime duration(int64_t start,uint32_t seconds,uint32_t reserve){
        if(start<0||!seconds||!reserve||start>INT64_MAX-(int64_t(seconds)+reserve)*1000)throw std::invalid_argument("cycle duration contract");
        return {start,start+int64_t(seconds)*1000,start+(int64_t(seconds)+reserve)*1000};
    }
    static CycleTime resume(int64_t now,uint32_t seconds,uint32_t reserve,int64_t completed){
        auto t=duration(now,seconds,reserve);if(completed<0||completed>int64_t(seconds)*1000)throw std::invalid_argument("invalid completed learning time");
        t.work_end_ms-=completed;t.end_ms-=completed;return t;
    }
    int64_t completed(int64_t now,int64_t previous,uint32_t seconds)const{return std::min(int64_t(seconds)*1000,previous+std::max(int64_t(0),now-start_ms));}
    bool finishing(int64_t now)const{return end_ms&&now>=work_end_ms;}
    bool expired(int64_t now)const{return end_ms&&now>=end_ms;}
};
inline uint32_t update_target(uint32_t step,uint32_t count){if(count>UINT32_MAX-step)throw std::overflow_error("optimizer u32 step counter exhausted");return step+count;}
}
