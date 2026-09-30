#include "neural_update_cuda.hpp"
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
struct Forward {
    cublasLtMatmulDesc_t operation{};cublasLtMatrixLayout_t a{},b{},d{};cublasLtMatmulAlgo_t algorithm{};bool fused{};
    ~Forward(){if(operation)cublasLtMatmulDescDestroy(operation);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(d)cublasLtMatrixLayoutDestroy(d);}
};
}
struct FusedMlp {
    FusedMlpBuffers buffers;cublasHandle_t gemm{};cublasLtHandle_t lt{};Forward forward[3];
    explicit FusedMlp(const FusedMlpBuffers& b):buffers(b){}
    ~FusedMlp(){if(gemm)cublasDestroy(gemm);if(lt)cublasLtDestroy(lt);}
    void stream(cudaStream_t s){blas(cublasSetStream(gemm,s));blas(cublasSetWorkspace(gemm,buffers.workspace,buffers.workspace_bytes));}
    void multiply(cublasOperation_t a,cublasOperation_t b,int m,int n,int k,const float* x,int ldx,const float* y,int ldy,float* z,int ldz){
        const float one=1,zero=0;blas(cublasGemmEx(gemm,a,b,m,n,k,&one,x,CUDA_R_32F,ldx,y,CUDA_R_32F,ldy,&zero,z,CUDA_R_32F,ldz,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    }
};
FusedMlp* create_fused_mlp(const FusedMlpBuffers& b){
    if(!b.rows||b.rows>65536||!b.workspace||b.workspace_bytes<4096)throw std::invalid_argument("fused MLP storage contract");
    auto* p=new FusedMlp(b);
    try{blas(cublasCreate(&p->gemm));blas(cublasSetMathMode(p->gemm,CUBLAS_PEDANTIC_MATH));blas(cublasSetAtomicsMode(p->gemm,CUBLAS_ATOMICS_NOT_ALLOWED));blas(cublasLtCreate(&p->lt));
        for(unsigned l=0;l<3;++l){auto& f=p->forward[l];int in=l?64:128,out=l==2?12:64;
            blas(cublasLtMatmulDescCreate(&f.operation,CUBLAS_COMPUTE_32F_PEDANTIC,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;
            blas(cublasLtMatmulDescSetAttribute(f.operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
            auto epilogue=l==2?CUBLASLT_EPILOGUE_BIAS:CUBLASLT_EPILOGUE_RELU_BIAS;
            blas(cublasLtMatmulDescSetAttribute(f.operation,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
            blas(cublasLtMatmulDescSetAttribute(f.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&b.parameter[l*2+1],sizeof(float*)));
            // Column-major views of the existing contiguous row-major tensors.
            // Epilogues do not require transposing/copying those tensors.
            blas(cublasLtMatrixLayoutCreate(&f.a,CUDA_R_32F,in,out,in));blas(cublasLtMatrixLayoutCreate(&f.b,CUDA_R_32F,in,b.rows,in));blas(cublasLtMatrixLayoutCreate(&f.d,CUDA_R_32F,out,b.rows,out));
            cublasLtMatmulPreference_t preference{};blas(cublasLtMatmulPreferenceCreate(&preference));
            auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&b.workspace_bytes,sizeof(b.workspace_bytes));
            cublasLtMatmulHeuristicResult_t result{};int count=0;if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulAlgoGetHeuristic(p->lt,f.operation,f.a,f.b,f.d,f.d,preference,1,&result,&count);cublasLtMatmulPreferenceDestroy(preference);
            if(status==CUBLAS_STATUS_SUCCESS&&count&&result.state==CUBLAS_STATUS_SUCCESS){f.algorithm=result.algo;f.fused=true;}
            else if(status!=CUBLAS_STATUS_SUCCESS&&status!=CUBLAS_STATUS_NOT_SUPPORTED)blas(status);
        }
        return p;
    }catch(...){delete p;throw;}
}
void destroy_fused_mlp(FusedMlp* p) noexcept{delete p;}
uint32_t fused_mlp_epilogues(const FusedMlp* p){return unsigned(p->forward[0].fused)|unsigned(p->forward[1].fused)<<1|unsigned(p->forward[2].fused)<<2;}
std::array<int32_t,3> fused_mlp_algorithms(const FusedMlp* p){std::array<int32_t,3> ids{-1,-1,-1};for(unsigned i=0;i<3;++i)if(p->forward[i].fused){size_t written=0;blas(cublasLtMatmulAlgoConfigGetAttribute(&p->forward[i].algorithm,CUBLASLT_ALGO_CONFIG_ID,&ids[i],sizeof(ids[i]),&written));}return ids;}
void fused_mlp_forward(FusedMlp* p,cudaStream_t stream){auto& b=p->buffers;p->stream(stream);
    for(unsigned l=0;l<3;++l){int in=l?64:128,out=l==2?12:64;auto* x=l?b.hidden[l-1]:b.input;auto* y=l==2?b.output:b.hidden[l];auto& f=p->forward[l];
        if(f.fused){const float one=1,zero=0;blas(cublasLtMatmul(p->lt,f.operation,&one,b.parameter[l*2],f.a,x,f.b,&zero,y,f.d,y,f.d,&f.algorithm,b.workspace,b.workspace_bytes,stream));}
        else {p->multiply(CUBLAS_OP_T,CUBLAS_OP_N,out,b.rows,in,b.parameter[l*2],in,x,in,y,out);bias_activation<<<(b.rows*out+255)/256,256,0,stream>>>(y,b.parameter[l*2+1],b.rows,out,l<2);}
    }
}
void fused_mlp_backward(FusedMlp* p,cudaStream_t stream){auto& b=p->buffers;p->stream(stream);
    for(int l=2;l>=0;--l){int in=l?64:128,out=l==2?12:64;auto* x=l?b.hidden[l-1]:b.input;auto* dy=l==2?b.derivative:b.delta[l];
        derivative_bias<<<(out+7)/8,256,0,stream>>>(dy,l==2?nullptr:b.hidden[l],b.gradient[l*2+1],b.rows,out);
        p->multiply(CUBLAS_OP_N,CUBLAS_OP_T,in,out,b.rows,x,in,dy,out,b.gradient[l*2],in);
        if(l)p->multiply(CUBLAS_OP_N,CUBLAS_OP_N,in,b.rows,out,b.parameter[l*2],in,dy,out,b.delta[l-1],in);
    }
}
}
