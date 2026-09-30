#pragma once
#include "blitz/mesh.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define BLITZ_VERTEX_HD __host__ __device__
#else
#define BLITZ_VERTEX_HD
#endif
namespace blitz::neural {
// Separate streams: position 6, UV 4, color 4, normal 4, tangent 4 bytes.
// Bounds are per mesh, never per camera. Zero extent encodes as zero.
struct VertexBounds { Vec3 low{}, extent{}; };
BLITZ_VERTEX_HD inline uint16_t pack_unorm16(float value,float low,float extent) {
    if(extent==0)return 0;
    double q=(double(value)-low)/extent*65535.;
    return uint16_t(q<=0?0:q>=65535?65535:floor(q+.5));
}
BLITZ_VERTEX_HD inline float unpack_unorm16(uint16_t value,float low,float extent) {
    return low+extent*(float(value)/65535.f);
}
BLITZ_VERTEX_HD inline uint32_t pack_direction(Vec3 v,int alpha=0) {
    double length=sqrt(double(v.x)*v.x+double(v.y)*v.y+double(v.z)*v.z);
    uint32_t result=(uint32_t(alpha)&3u)<<30;
    if(length==0)return result;
    const float values[3]={v.x,v.y,v.z};
    for(unsigned i=0;i<3;++i){double x=double(values[i])/length*511.;int32_t q=int32_t(x<0?ceil(x-.5):floor(x+.5));result|=(uint32_t(q)&1023u)<<(10*i);}
    return result;
}
BLITZ_VERTEX_HD inline Vec3 unpack_direction(uint32_t packed) {
    float v[3];for(unsigned i=0;i<3;++i){int32_t x=int32_t((packed>>(10*i))&1023u);if(x&512)x-=1024;v[i]=float(x)/511.f;}
    float length=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    return length>0?Vec3{v[0]/length,v[1]/length,v[2]/length}:Vec3{};
}
inline VertexBounds vertex_bounds(MeshView mesh) {
    if(!mesh.positions.count)return {};
    Vec3 low=mesh.positions[0],high=low;
    for(size_t i=1;i<mesh.positions.count;++i){auto p=mesh.positions[i];low={std::min(low.x,p.x),std::min(low.y,p.y),std::min(low.z,p.z)};high={std::max(high.x,p.x),std::max(high.y,p.y),std::max(high.z,p.z)};}
    auto extent=[](float lo,float hi){double d=double(hi)-lo;float f=float(d);return double(f)<d?std::nextafter(f,INFINITY):f;};
    return {low,{extent(low.x,high.x),extent(low.y,high.y),extent(low.z,high.z)}};
}
}
#undef BLITZ_VERTEX_HD
