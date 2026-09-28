#pragma once
#include "neural_internal.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <utility>
namespace blitz::neural::gpu {
inline void check(cudaError_t e) {if(e==cudaErrorMemoryAllocation)throw std::length_error("CUDA allocation exceeds available device memory");if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e));}
inline void check(cublasStatus_t e) {if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS failure "+std::to_string(int(e)));}
struct Device {
    int previous{};size_t limit{},live{},peak{};
    explicit Device(const NeuralOptions& o):limit(size_t(o.memory_mib)*1024*1024) {
        if(o.memory_mib<128||o.memory_mib>65536||o.device<0)throw std::invalid_argument("invalid CUDA options");
        check(cudaGetDevice(&previous));check(cudaSetDevice(o.device));
    }
    ~Device(){cudaSetDevice(previous);}
    Device(const Device&)=delete;
};
template<class T> struct Buffer {
    Device* device{};T* p{};size_t n{};
    Buffer()=default;
    Buffer(Device& d,size_t count):device(&d),n(count) {
        if(n>SIZE_MAX/sizeof(T)||n*sizeof(T)>d.limit-d.live)throw std::length_error("CUDA workspace exceeds configured memory cap");
        if(n)check(cudaMalloc(&p,n*sizeof(T)));d.live+=n*sizeof(T);d.peak=std::max(d.peak,d.live);
    }
    ~Buffer(){if(p){cudaFree(p);device->live-=n*sizeof(T);}}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& b) noexcept:device(b.device),p(std::exchange(b.p,nullptr)),n(b.n){}
    void zero(){if(n)check(cudaMemset(p,0,n*sizeof(T)));}
    void upload(std::span<const T> a){if(a.size()!=n)throw std::invalid_argument("CUDA upload size");if(n)check(cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> download()const{std::vector<T> a(n);if(n)check(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}
};
inline unsigned blocks(size_t n){return unsigned((n+255)/256);}
struct Blas {cublasHandle_t value{};Blas(){check(cublasCreate(&value));}~Blas(){cublasDestroy(value);}operator cublasHandle_t()const{return value;}};
}
