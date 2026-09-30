#include "neural_action_data.hpp"
#include "neural_mesh_cache.hpp"
#include <sstream>
#include <iostream>
using namespace blitz::neural;using namespace blitz::neural::training;
using namespace blitz;
static void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
int main(){try{
    {
        Mesh source;source.positions={{-1,-2,0},{2,-2,0},{0,3,0}};source.normals.assign(3,{0,0,1});source.tangents.assign(3,{1,0,0,-1});source.uv={{-8,8},{0,0},{8,-8}};source.colors.assign(3,{0,17,255,73});source.indices={0,1,2};source.materials={3};source.double_sided={0,0,0,1};
        std::stringstream f(std::ios::in|std::ios::out|std::ios::binary);write_reference(f,source);f.seekg(0);auto reference=read_reference(f);require(same_mesh_data(source.view(),reference.view()),"reference cache changes immutable source");
        std::stringstream packed(std::ios::in|std::ios::out|std::ios::binary);write_packed_mesh(packed,source.view());packed.seekg(0);auto [draw,domain]=read_packed_mesh(packed);require(draw.indices==source.indices&&draw.materials==source.materials&&draw.double_sided==source.double_sided,"packed cache topology/material changes");
        for(size_t i=0;i<3;++i){require(draw.colors[i].a==73&&draw.colors[i].b==255&&draw.tangents[i].w==-1,"packed cache attribute/sign changes");require(std::abs(draw.positions[i].x-source.positions[i].x)<=3.f/131070+1e-6,"packed cache position error bound");}
        std::stringstream again(std::ios::in|std::ios::out|std::ios::binary);write_packed_mesh(again,draw.view());require(again.str()==packed.str(),"decode/repack changes cached draw codes");
        source.uv[0].x=std::nextafter(8.f,INFINITY);bool rejected=false;try{write_packed_mesh(again,source.view());}catch(const std::invalid_argument&){rejected=true;}require(rejected,"packed cache silently clamps UV");
    }
    for(uint32_t h=0;h<65536;++h){if((h&0x7c00)==0x7c00)continue;float f=half_float(uint16_t(h));_Float16 reference=std::bit_cast<_Float16>(uint16_t(h));require(f==float(reference),"half decode differs from compiler IEEE conversion");require(half_bits(std::bit_cast<uint32_t>(f))==h,"half finite round trip");}
    for(float x:{0.f,-0.f,1.f,-1.f,65504.f,65520.f,1e-20f,0x1p-24f,0x1p-25f,1.00048828125f,1.00146484375f})require(half_bits(std::bit_cast<uint32_t>(x))==std::bit_cast<uint16_t>(_Float16(x)),"half nearest-even boundary");
    for(auto architecture:{action_schema,placement_schema}){
        ActionData a;a.architecture=architecture;auto width=policy_inputs(architecture);
        for(unsigned s=0;s<3;++s){for(unsigned r=0;r<4;++r){size_t at=a.x.size();a.x.resize(at+width);for(unsigned c=0;c<width;++c){auto slot=feature_slot(c);float x=slot< -32?float(c+s)/80:slot<0?float((c+r)%2):float(int((c*17+r*3)%41)-20)/23;
                    if(slot>=0&&feature_encoding(c)==FeatureEncoding::Color)x=float((c+r*71)%256)/255.f;
                    if(slot>=0&&feature_encoding(c)==FeatureEncoding::Unorm)x=float((c+r)%13)/12;
                    a.x[at+c]=x;}
                a.x[at+78]=a.x[at+23];a.x[at+79]=a.x[at+47];
                // Half range exceptions, normalized-range exceptions, byte-color
                // exceptions and exact endpoints coexist in the same page.
                a.x[at]=r==0?1e-20f:r==1?1e20f:r==2?-65504.f:0.f;a.x[at+3]=r==0?1.1f:-1.f;a.x[at+6]=r==0?.12345f:1.f;
                a.labels.push_back(architecture==placement_schema?(r==0?255:r==1?59:r==2?24:8):uint8_t(r==0?15:8));a.from.push_back(r);a.to.push_back(r+1);
                if(architecture==placement_schema)for(unsigned c=0;c<9;++c)a.targets.push_back((a.labels.back()&(32u<<(c/3)))?float(int(c)-4)/8:0);
            }a.offsets.push_back(uint32_t(a.labels.size()));a.progress.push_back(float(s)/2);}
        auto packed=compact_actions(a);validate_compact(packed);auto b=expand_compact(packed);require(a.labels==b.labels&&a.offsets==b.offsets&&a.from==b.from&&a.to==b.to,"compact labels/IDs changed");
        for(size_t i=0;i<a.x.size();++i){unsigned c=i%width;float x=a.x[i],y=b.x[i];auto kind=feature_encoding(c);float tolerance=kind==FeatureEncoding::Half?std::abs(x)/1024+0x1p-24f:kind==FeatureEncoding::Snorm?1.f/65534:kind==FeatureEncoding::Unorm?1.f/131070:0;require(std::abs(x-y)<=tolerance+1e-7f,"feature quantization bound");if(feature_slot(c)<0||std::abs(x)>65504||std::abs(x)<0x1p-25f)require(x==y,"exact feature/exception changed");}
        for(size_t i=0;i<a.targets.size();++i)require(std::abs(a.targets[i]-b.targets[i])<=(i%9<3?1.f:2.f)/65534+1e-7f,"target quantization bound");
        std::stringstream f(std::ios::in|std::ios::out|std::ios::binary);write_compact(f,packed);f.seekg(8);auto restored=read_compact(f);require(restored.values==packed.values&&restored.escape_values==packed.escape_values&&restored.targets==packed.targets,"compact serialization differs");
        auto invalid=packed;invalid.escape_ids.push_back(0);invalid.escape_values.push_back(1);bool rejected=false;try{validate_compact(invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"unordered escape table accepted");
        invalid=packed;invalid.target_offsets.back()++;rejected=false;try{validate_compact(invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"bad target offsets accepted");
        invalid=packed;invalid.escape_ids.clear();invalid.escape_values.clear();rejected=false;try{validate_compact(invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"missing feature exceptions accepted");
        auto path=fs::temp_directory_path()/("blitz-compact-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));save_actions(path,a,false);auto legacy=load_actions(path);require(legacy.x==a.x&&legacy.targets==a.targets,"legacy action shard changed");auto transcoded=load_compact_actions(path);require(expand_compact(transcoded).x==b.x,"legacy direct compact conversion differs");save_actions(path,a);require(load_actions(path).x==b.x,"default compact shard reader differs");fs::remove(path);
    }
    std::cout<<"compact action storage contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
