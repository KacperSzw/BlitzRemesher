#include "neural_update_cuda.hpp"
#include "neural_linear.hpp"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>

namespace blitz::neural::training {
namespace {
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("FP32 MLP BLAS status "+std::to_string(int(s)));}
__global__ void bias_activation(float* x,const float* bias,uint32_t rows,uint32_t width,bool relu){
    auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*width)return;float v=x[i]+bias[i%width];x[i]=relu&&v<0?0:v;
}
// Eight adjacent channels per block make the row-major reads/writes coalesced.
// Warp and block reductions stay deterministic; no atomics or zero-fill.
__global__ void derivative_bias(float* dy,const float* activation,float* bias,uint32_t rows,uint32_t width){
    uint32_t channel=blockIdx.x*8+threadIdx.x%8;float sum=0;
    if(channel<width)for(uint32_t row=threadIdx.x/8;row<rows;row+=32){auto i=size_t(row)*width+channel;float g=dy[i];if(activation&&activation[i]<=0)g=0;dy[i]=g;sum+=g;}
    sum+=__shfl_down_sync(0xffffffffu,sum,16);sum+=__shfl_down_sync(0xffffffffu,sum,8);
    __shared__ float partial[64];if((threadIdx.x&31)<8)partial[(threadIdx.x/32)*8+threadIdx.x%8]=sum;__syncthreads();
    if(threadIdx.x<8&&channel<width){sum=0;for(unsigned warp=0;warp<8;++warp)sum+=partial[warp*8+threadIdx.x];bias[channel]=sum;}
}

}
struct FusedMlp {
    FusedMlpBuffers buffers;cublasHandle_t gemm{};
    explicit FusedMlp(const FusedMlpBuffers& b):buffers(b){}
    ~FusedMlp(){if(gemm)cublasDestroy(gemm);}
    void stream(cudaStream_t s){blas(cublasSetStream(gemm,s));blas(cublasSetWorkspace(gemm,buffers.workspace,buffers.workspace_bytes));}
    void multiply(cublasOperation_t a,cublasOperation_t b,int m,int n,int k,const float* x,int ldx,const float* y,int ldy,float* z,int ldz){
        const float one=1,zero=0;blas(cublasGemmEx(gemm,a,b,m,n,k,&one,x,CUDA_R_32F,ldx,y,CUDA_R_32F,ldy,&zero,z,CUDA_R_32F,ldz,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    }
};
FusedMlp* create_fused_mlp(const FusedMlpBuffers& b){
    if((b.width!=64&&b.width!=128&&b.width!=256)||!b.rows||b.rows>65536||!b.workspace||b.workspace_bytes<4096)throw std::invalid_argument("fused MLP storage contract");
    auto* p=new FusedMlp(b);
    try{blas(cublasCreate(&p->gemm));blas(cublasSetMathMode(p->gemm,CUBLAS_PEDANTIC_MATH));blas(cublasSetAtomicsMode(p->gemm,CUBLAS_ATOMICS_NOT_ALLOWED));        return p;
    }catch(...){delete p;throw;}
}
void destroy_fused_mlp(FusedMlp* p) noexcept{delete p;}
uint32_t fused_mlp_epilogues(const FusedMlp*){return 7;}
std::array<int32_t,3> fused_mlp_algorithms(const FusedMlp*){return {-2,-2,-2};} // Compensated FP32, not cuBLAS algorithm IDs.
void fused_mlp_forward(FusedMlp* p,cudaStream_t stream){auto& b=p->buffers;
    for(unsigned l=0;l<3;++l){int in=l?b.width:128,out=l==2?12:b.width;auto* x=l?b.hidden[l-1]:b.input;auto* y=l==2?b.output:b.hidden[l];compensated_linear(x,b.parameter[l*2],b.parameter[l*2+1],y,b.rows,in,out,l<2,stream);}
}
void fused_mlp_backward(FusedMlp* p,cudaStream_t stream){auto& b=p->buffers;p->stream(stream);
    for(int l=2;l>=0;--l){int in=l?b.width:128,out=l==2?12:b.width;auto* x=l?b.hidden[l-1]:b.input;auto* dy=l==2?b.derivative:b.delta[l];
        derivative_bias<<<(out+7)/8,256,0,stream>>>(dy,l==2?nullptr:b.hidden[l],b.gradient[l*2+1],b.rows,out);
        p->multiply(CUBLAS_OP_N,CUBLAS_OP_T,in,out,b.rows,x,in,dy,out,b.gradient[l*2],in);
        if(l)p->multiply(CUBLAS_OP_N,CUBLAS_OP_N,in,b.rows,out,b.parameter[l*2],in,dy,out,b.delta[l-1],in);
    }
}
}
