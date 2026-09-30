#include "neural_quantization.hpp"
#include <set>
namespace blitz::neural {
namespace {
double cost(const Pixel& a,const Pixel& b,double spatial,const EvalSettings& e){
    if(!b.visible)return INFINITY;double result=spatial;
    if(e.profile!=Profile::Coverage&&e.weights.normal>0){double angle=std::acos(std::clamp(dot(a.normal,b.normal)/std::max(1e-30,length(a.normal)*length(b.normal)),-1.,1.));result+=std::pow(angle*e.weights.normal,2);}
    if(e.profile==Profile::Attributes){auto d=Vec3{a.color.x-b.color.x,a.color.y-b.color.y,a.color.z-b.color.z};result+=e.weights.color*e.weights.color*dot(d,d);if(a.material!=b.material)result+=e.weights.material*e.weights.material;}return result;
}
struct Witnesses {uint64_t count{};std::set<uint32_t> faces;};
Witnesses failures(const DiagnosticRaster& a,const DiagnosticRaster& b,const EvalSettings& e){
    Witnesses out;int size=int(a.raster.width),radius=int(std::ceil(e.limit*e.supersample));double inverse=1./(e.supersample*e.supersample),limit=e.limit*e.limit;
    for(unsigned direction=0;direction<2;++direction){const auto& from=direction?b:a;const auto& to=direction?a:b;
        for(size_t i=0;i<from.raster.pixels.size();++i){const auto& p=from.raster.pixels[i];if(!p.visible)continue;double best=cost(p,to.raster.pixels[i],0,e);if(best<=limit)continue;int x=int(i%size),y=int(i/size);
            for(int yy=std::max(0,y-radius);yy<=std::min(size-1,y+radius);++yy)for(int xx=std::max(0,x-radius);xx<=std::min(size-1,x+radius);++xx){double spatial=((x-xx)*(x-xx)+(y-yy)*(y-yy))*inverse;if(spatial>limit||spatial>=best)continue;best=std::min(best,cost(p,to.raster.pixels[size_t(yy)*size+xx],spatial,e));}
            if(best<=limit)continue;++out.count;if(from.faces[i]!=UINT32_MAX)out.faces.insert(from.faces[i]);if(to.faces[i]!=UINT32_MAX)out.faces.insert(to.faces[i]);
        }
    }return out;
}
}
PackingRepairResult repair_packing_gpu(MeshView source,const NeuralOptions& options,const EvalSettings& primary,uint32_t budget,MeshView prepared,const EvalSettings* additional){
    if(prepared.positions.count&&(source.positions.count!=prepared.positions.count||source.indices.size()!=prepared.indices.size()||!std::equal(source.indices.begin(),source.indices.end(),prepared.indices.begin())))throw std::invalid_argument("packing repair requires source topology and IDs");
    PackingRepairResult out;out.mesh=copy_mesh(prepared.positions.count?prepared:source);AuditCuda audit(options,source);auto b=bounds(source);unsigned failed_check=0;
    auto evaluate=[&]{auto first=audit.evaluate(source,out.mesh.view(),b,primary);failed_check=0;if(first.passed&&additional){auto second=audit.evaluate(source,out.mesh.view(),b,*additional);if(!second.passed){failed_check=1;return second;}}return first;};out.initial=out.final=evaluate();
    if(out.final.passed||options.draw_storage()==NeuralVertexStorage::Float32||out.final.resource_limited)return out;
    auto domain=vertex_bounds(source);auto control=options;control.vertex_storage=NeuralVertexStorage::Float32;std::set<uint32_t> fixed;
    while(!out.final.passed&&out.trials<budget&&!(primary.cancelled&&primary.cancelled())){const auto& e=failed_check?*additional:primary;unsigned check_before=failed_check;auto views=cameras(b,e.screen_size,e.views);
        uint32_t v=out.final.worst_view;auto a=diagnostic_raster(source,b,views[v],e.screen_size,e.supersample,e.force_two_sided,control);auto before=diagnostic_raster(out.mesh.view(),b,views[v],e.screen_size,e.supersample,e.force_two_sided,options,&domain);auto witnesses=failures(a,before,e);if(!witnesses.count)break;
        std::set<uint32_t> vertices;for(auto face:witnesses.faces)for(unsigned j=0;j<3;++j)vertices.insert(source.indices[size_t(face)*3+j]);bool improved=false;
        for(int radius:{1,2,4,8})for(auto vertex:vertices){if(improved||out.trials>=budget)break;Vec3 old=out.mesh.positions[vertex];std::vector<uint32_t> coupled;for(uint32_t i=0;i<source.positions.count;++i){auto p=out.mesh.positions[i];if(p.x==old.x&&p.y==old.y&&p.z==old.z)coupled.push_back(i);}
            for(unsigned axis=0;axis<3&&!improved&&out.trials<budget;++axis)for(int sign:{-1,1}){if(improved||out.trials>=budget||(e.cancelled&&e.cancelled()))break;float value=axis==0?old.x:axis==1?old.y:old.z,lo=axis==0?domain.low.x:axis==1?domain.low.y:domain.low.z,span=axis==0?domain.extent.x:axis==1?domain.extent.y:domain.extent.z;
                int direction=sign*radius,code=int(pack_unorm16(value,lo,span))+direction;if(!span||code<0||code>65535)continue;Vec3 next=old;(axis==0?next.x:axis==1?next.y:next.z)=unpack_unorm16(uint16_t(code),lo,span);for(auto id:coupled)out.mesh.positions[id]=next;
                bool legal=true;for(size_t f=0;f<source.triangles()&&legal;++f){uint32_t ids[]={source.indices[f*3],source.indices[f*3+1],source.indices[f*3+2]};bool touched=false;for(auto id:ids)touched|=std::find(coupled.begin(),coupled.end(),id)!=coupled.end();if(!touched)continue;auto n=cross(source.positions[ids[1]]-source.positions[ids[0]],source.positions[ids[2]]-source.positions[ids[0]]),q=cross(out.mesh.positions[ids[1]]-out.mesh.positions[ids[0]],out.mesh.positions[ids[2]]-out.mesh.positions[ids[0]]);legal=length(q)>1e-15&&dot(n,q)>.05*length(n)*length(q);}
                if(legal){++out.trials;auto raster=diagnostic_raster(out.mesh.view(),b,views[v],e.screen_size,e.supersample,e.force_two_sided,options,&domain);auto after=failures(a,raster,e);
                    if(after.count<witnesses.count){auto measured=evaluate();if(!measured.resource_limited&&(measured.passed||failed_check>check_before||(failed_check==check_before&&measured.worst_view>=v))){out.final=measured;improved=true;fixed.insert(coupled.begin(),coupled.end());}else failed_check=check_before;}
                    out.attempts.push_back({out.trials,v,vertex,axis,direction,witnesses.count,after.count,improved});
                }if(!improved)for(auto id:coupled)out.mesh.positions[id]=old;
            }
        }if(!improved)break;
    }
    out.changed_vertices=uint32_t(fixed.size());return out;
}
}
