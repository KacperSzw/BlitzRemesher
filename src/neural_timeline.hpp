#pragma once
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#endif
namespace blitz::neural {
struct Timeline {
    explicit Timeline(const char* name){
#if __has_include(<nvtx3/nvToolsExt.h>)
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~Timeline(){
#if __has_include(<nvtx3/nvToolsExt.h>)
        nvtxRangePop();
#endif
    }
};
}
