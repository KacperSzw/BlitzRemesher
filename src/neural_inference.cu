#include "neural_cuda.cuh"
#include "neural_numeric.hpp"
#include "neural_placement.hpp"
namespace blitz {
bool neural_available(int32_t device) noexcept {int count=0;auto e=cudaGetDeviceCount(&count);if(e!=cudaSuccess){cudaGetLastError();return false;}return device>=0&&device<count;}
namespace neural {
using namespace gpu;
namespace {
__global__ void aggregate(const float* in,float* out,const uint32_t* offsets,const uint32_t* neighbors,uint32_t n,uint32_t width) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*width)return;
    auto vertex=uint32_t(i/width),channel=uint32_t(i%width),first=offsets[vertex],end=offsets[vertex+1];float sum=0;
    for(auto k=first;k<end;++k)sum+=in[size_t(neighbors[k])*width+channel];
    out[size_t(vertex)*width*2+channel]=in[i];out[size_t(vertex)*width*2+width+channel]=sum/float(max(1u,end-first));
}
__global__ void activate(float* x,const float* bias,uint32_t n,uint32_t width,bool relu) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*width)return;float v=x[i]+(bias?bias[i%width]:0);x[i]=relu?fmaxf(0,v):v;
}
__global__ void concatenate(const float* x,const float* c,float* out,uint32_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*(hidden+conditions))return;auto channel=i%(hidden+conditions);out[i]=channel<hidden?x[i/(hidden+conditions)*hidden+channel]:c[channel-hidden];
}
void capture(NumericTrace* trace,const std::string& name,const float* data,uint32_t n,uint32_t width) {
    if(!trace)return;std::vector<float> host(size_t(n)*width);check(cudaMemcpy(host.data(),data,host.size()*sizeof(float),cudaMemcpyDeviceToHost));
    trace->push_back({name,width,{host.begin(),host.end()}});
}
void linear(cublasHandle_t handle,const float* x,const float* weights,float* y,uint32_t n,int in,int out,bool relu,NumericTrace* trace=nullptr,unsigned layer=0) {
    const auto prefix="layer"+std::to_string(layer)+"/";capture(trace,prefix+"input",x,n,in);
    float alpha=1,beta=0;check(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,int(n),in,&alpha,weights,in,x,in,&beta,y,out));
    capture(trace,prefix+"gemm",y,n,out);
    activate<<<blocks(size_t(n)*out),256>>>(y,weights+size_t(in)*out,n,out,trace?false:relu);check(cudaGetLastError());
    if(trace){capture(trace,prefix+"bias",y,n,out);if(relu){activate<<<blocks(size_t(n)*out),256>>>(y,nullptr,n,out,true);check(cudaGetLastError());}}
    capture(trace,prefix+"output",y,n,out);
}
}
std::vector<float> encode_cuda(const Graph& g,const WeightsData& w,const NeuralOptions& options,NumericTrace* trace) {
    validate_graph(g);
    if(g.size()>262144||w.values.size()!=weight_count||g.x.size()!=g.size()*features||g.offsets.size()!=g.size()+1)throw std::invalid_argument("invalid CUDA graph dimensions");
    if(g.size()==0)return {};Device d(options);Blas blas;uint32_t n=uint32_t(g.size());
    Buffer<uint32_t> offsets(d,g.offsets.size()),neighbors(d,g.neighbors.size());offsets.upload(g.offsets);neighbors.upload(g.neighbors);
    Buffer<float> weights(d,w.values.size()),input(d,g.x.size()),a(d,size_t(n)*hidden),b(d,size_t(n)*hidden),cat(d,size_t(n)*hidden*2);
    weights.upload(w.values);input.upload(g.x);const float* x=input.p;float* y=a.p;size_t at=0;
    for(unsigned layer=0;layer<3;++layer) {
        uint32_t width=layer==0?features:hidden;aggregate<<<blocks(size_t(n)*width),256>>>(x,cat.p,offsets.p,neighbors.p,n,width);check(cudaGetLastError());
        linear(blas,cat.p,weights.p+at,y,n,width*2,hidden,true,trace,layer);at+=size_t(hidden)*(width*2+1);x=y;y=y==a.p?b.p:a.p;
    }
    check(cudaDeviceSynchronize());return a.download();
}
Prediction predict_cuda(std::span<const float> embedding,const std::array<float,conditions>& condition,const WeightsData& w,const NeuralOptions& options,NumericTrace* trace) {
    if(embedding.size()%hidden||w.values.size()!=weight_count)throw std::invalid_argument("invalid embedding dimensions");
    Prediction result;result.values.resize(embedding.size()/hidden*outputs);Device d(options);Blas blas;
    Buffer<float> weights(d,w.values.size()),c(d,conditions);weights.upload(w.values);c.upload(condition);
    size_t at=0;for(unsigned i=0;i<3;++i)at+=size_t(layer_out[i])*(layer_in[i]+1);
    for(size_t first=0;first<embedding.size()/hidden;first+=65536) {
        uint32_t n=uint32_t(std::min<size_t>(65536,embedding.size()/hidden-first));
        Buffer<float> input(d,size_t(n)*hidden),cat(d,size_t(n)*(hidden+conditions)),h(d,size_t(n)*hidden),out(d,size_t(n)*outputs);
        input.upload(embedding.subspan(first*hidden,size_t(n)*hidden));
        concatenate<<<blocks(cat.n),256>>>(input.p,c.p,cat.p,n);check(cudaGetLastError());
        linear(blas,cat.p,weights.p+at,h.p,n,hidden+conditions,hidden,true,trace,3);
        linear(blas,h.p,weights.p+at+hidden*(hidden+conditions+1),out.p,n,hidden,outputs,false,trace,4);
        check(cudaMemcpy(result.values.data()+first*outputs,out.p,out.n*sizeof(float),cudaMemcpyDeviceToHost));
    }
    return result;
}
struct ActionCuda::Impl {
    Device device;Blas blas;Buffer<float> weights,input,a,b,output;uint32_t batch,architecture;int id;
    Impl(const WeightsData& w,const NeuralOptions& options,uint32_t count):device(options),weights(device,w.values.size()),input(device,size_t(count)*policy_inputs(w.architecture)),a(device,size_t(count)*hidden),b(device,size_t(count)*hidden),output(device,size_t(count)*policy_outputs(w.architecture)),batch(count),architecture(w.architecture),id(options.device) {
        weights.upload(w.values);check(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
        check(cudaSetDevice(device.previous));
    }
    ~Impl(){cudaSetDevice(id);}
};
ActionCuda::ActionCuda(const WeightsData& w,const NeuralOptions& options,uint32_t count) {
    if(!policy_weights(w.architecture)||w.values.size()!=policy_weights(w.architecture)||count<1||count>65536)throw std::invalid_argument("invalid action model or batch dimensions");
    for(float v:w.values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite action weights");
    impl_=std::make_unique<Impl>(w,options,count);
}
ActionCuda::~ActionCuda(){if(impl_){int previous=0;cudaGetDevice(&previous);impl_.reset();cudaSetDevice(previous);}}
uint32_t ActionCuda::architecture()const{return impl_->architecture;}
void ActionCuda::predict_device(const float* input,float* output,uint32_t rows) {
    auto& p=*impl_;NeuralOptions options;options.device=p.id;Device guard(options);
    auto width=policy_inputs(p.architecture),outputs=policy_outputs(p.architecture);
    for(uint32_t row=0;row<rows;row+=p.batch){auto n=std::min(p.batch,rows-row);const float* in=input+size_t(row)*width;size_t at=0;
        for(unsigned layer=0;layer<3;++layer){auto columns=layer?hidden:width,channels=layer==2?outputs:hidden;float* target=layer==0?p.a.p:layer==1?p.b.p:output+size_t(row)*outputs;
            linear(p.blas,in,p.weights.p+at,target,n,columns,channels,layer<2);in=target;at+=size_t(channels)*(columns+1);}}
}
std::vector<float> ActionCuda::predict(std::span<const float> x) {
    auto& p=*impl_;auto width=policy_inputs(p.architecture),outputs=policy_outputs(p.architecture);
    if(x.size()%width)throw std::invalid_argument("invalid action input dimensions");
    for(float v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite action input");
    NeuralOptions options;options.device=p.id;Device guard(options);std::vector<float> out(x.size()/width*outputs);
    for(size_t row=0;row<x.size()/width;row+=p.batch){auto n=uint32_t(std::min<size_t>(p.batch,x.size()/width-row));
        check(cudaMemcpy(p.input.p,x.data()+row*width,size_t(n)*width*sizeof(float),cudaMemcpyHostToDevice));
        const float* in=p.input.p;size_t at=0;
        for(unsigned layer=0;layer<3;++layer){auto columns=layer?hidden:width,channels=layer==2?outputs:hidden;float* target=layer==0?p.a.p:layer==1?p.b.p:p.output.p;linear(p.blas,in,p.weights.p+at,target,n,columns,channels,layer<2);in=target;at+=size_t(channels)*(columns+1);}
        check(cudaMemcpy(out.data()+row*outputs,p.output.p,size_t(n)*outputs*sizeof(float),cudaMemcpyDeviceToHost));}
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite action prediction");return out;
}
}
}
