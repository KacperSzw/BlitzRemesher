#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace blitz::neural {
// FP32 products and compensated sums; used by both training and deployment.
// ReLU and bias are fused. No FP64 or reduced-precision matrix arithmetic.
void compensated_linear(const float*,const float*,const float*,float*,uint32_t rows,uint32_t in,uint32_t out,bool relu,cudaStream_t);
}
