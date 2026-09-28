#include "blitz/remesher.hpp"
#include <stdexcept>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* bytes,size_t count) {
    if(count<10)return 0;blitz::Mesh m;size_t n=3+bytes[0]%30;
    for(size_t i=0;i<n;++i)m.positions.push_back({float(bytes[(i*3+1)%count])/255,float(bytes[(i*3+2)%count])/255,float(bytes[(i*3+3)%count])/255});
    if(bytes[6]&1)for(size_t i=0;i<n;++i)m.colors.push_back({bytes[(i*4)%count],bytes[(i*4+1)%count],bytes[(i*4+2)%count],bytes[(i*4+3)%count]});
    for(size_t i=1;i+2<count&&m.indices.size()<300;i+=3){m.indices.push_back(bytes[i]%n);m.indices.push_back(bytes[i+1]%n);m.indices.push_back(bytes[i+2]%n);}
    if(!blitz::validate(m.view()).empty())return 0;
    blitz::ReduceSettings s;s.output=blitz::OutputMode(bytes[1]%2);s.target_triangles=1+bytes[2]%20;
    s.coupled_wedges=bytes[3]%2;s.prune=bytes[4]%2;
    s.objective=blitz::Objective(bytes[5]%4);
    try{auto result=blitz::reduce(m.view(),s);if(!blitz::validate(result.view(m.view())).empty())__builtin_trap();}catch(const std::invalid_argument&){}
    return 0;
}
