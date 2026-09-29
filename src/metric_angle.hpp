#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
// Adapted from fdlibm e_acos.c (Sun, 1993), as retained by OpenLibm:
// https://github.com/JuliaMath/openlibm/blob/master/src/e_acos.c
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// CPU and CUDA libm acos differed by one ULP at an acceptance boundary. Use
// the same double operations for the sampled angular metric on both backends.
// Compile without contraction/fast math; inputs are clamped cosines in [-1,1].
#ifdef __CUDACC__
#define BLITZ_METRIC_HD __host__ __device__
#else
#define BLITZ_METRIC_HD
#endif
namespace blitz::detail {
BLITZ_METRIC_HD inline double metric_acos(double x) {
    constexpr double pi=3.14159265358979311600e+00,half=1.57079632679489655800e+00,lo=6.12323399573676603587e-17;
    if(x==1)return 0;if(x==-1)return pi+2*lo;
    const double z=x<-.5?(1+x)*.5:x>.5?(1-x)*.5:x*x;
    const double p=z*(1.66666666666666657415e-01+z*(-3.25565818622400915405e-01+z*(2.01212532134862925881e-01+z*(-4.00555345006794114027e-02+z*(7.91534994289814532176e-04+z*3.47933107596021167570e-05)))));
    const double q=1+z*(-2.40339491173441421878e+00+z*(2.02094576023350569471e+00+z*(-6.88283971605453293030e-01+z*7.70381505559019352791e-02)));
    const double r=p/q;
    if(x>=-.5&&x<=.5){if(x>-0x1p-57&&x<0x1p-57)return half+lo;return half-(x-(lo-x*r));}
    const double s=::sqrt(z);
    if(x<0)return pi-2*(s+(r*s-lo));
#ifdef __CUDA_ARCH__
    const double high=__hiloint2double(__double2hiint(s),0);
#else
    const double high=std::bit_cast<double>(std::bit_cast<uint64_t>(s)&0xffffffff00000000ull);
#endif
    const double correction=(z-high*high)/(s+high);
    return 2*(high+(r*s+correction));
}
}
#undef BLITZ_METRIC_HD
