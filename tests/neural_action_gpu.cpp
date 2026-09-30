#include "neural_action_gpu.hpp"
#include "neural_cuda.cuh"
#include <iostream>
using namespace blitz;using namespace blitz::neural;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
Mesh plane(unsigned n){Mesh m;for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){m.positions.push_back({float(x),float(y),0});m.normals.push_back({0,0,1});m.uv.push_back({float(x),float(y)});m.colors.push_back({uint8_t(x*21),uint8_t(y*12),100,255});m.tangents.push_back({1,0,0,1});}
    for(unsigned y=0;y+1<n;++y)for(unsigned x=0;x+1<n;++x){uint32_t a=y*n+x;m.indices.insert(m.indices.end(),{a,a+1,a+n,a+1,a+n+1,a+n});}m.double_sided={1};return m;}
void parity(const Mesh& mesh,unsigned iterations){NeuralOptions options;options.memory_mib=256;ActionState cpu(mesh.view());GpuActionState gpu(mesh.view(),options);auto before=copy_mesh(mesh.view());
    std::array<float,conditions> c{.7f,.1f,.2f,.8f,.5f,.3f,.05f,.4f};
    for(unsigned step=0;step<iterations;++step){auto a=cpu.actions(c),b=gpu.actions(c);if(a.size()!=b.size())throw std::runtime_error("action count "+std::to_string(a.size())+" != "+std::to_string(b.size()));
        for(size_t i=0;i<a.size();++i){require(a[i].action==b[i].action,"GPU action ordering or legality differs");for(unsigned j=0;j<action_features;++j)if(std::abs(a[i].x[j]-b[i].x[j])>2e-6)throw std::runtime_error("GPU feature "+std::to_string(j)+" differs: "+std::to_string(a[i].x[j])+" vs "+std::to_string(b[i].x[j]));}
        if(a.empty())break;auto action=a[(step*17+a.size()/2)%a.size()].action;auto expected=cpu.trial(action),actual=gpu.trial(action);require(expected.data.indices==actual.data.indices&&expected.data.materials==actual.data.materials,"GPU trial differs");
        require(gpu.view().faces==cpu.view().triangles(),"trial changed incumbent");cpu.commit(action);gpu.commit(action);require(gpu.view().faces==cpu.view().triangles(),"GPU commit count differs");
        bool rejected=false;try{gpu.trial(action);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"GPU executed stale action");
        // A rejected debug action must not poison the following valid operation.
        auto next=cpu.actions(c);if(!next.empty())gpu.trial(next.front().action);
    }require(same_mesh_data(mesh.view(),before.view()),"GPU action changed input");}
void executor(){auto m=plane(7);NeuralOptions options;options.memory_mib=256;EvalSettings e;e.screen_size=16;e.limit=4;e.supersample=2;e.max_supersample=4;e.views={2,1,819};
    GpuActionState gpu(m.view(),options);AuditCuda audit(options,m.view());
    for(auto ranking:{NeuralRanking::Constant,NeuralRanking::ShortestEdge,NeuralRanking::CurrentPlane})for(uint8_t batch:{uint8_t(1),uint8_t(4)})for(bool reject:{false,true}){
        std::vector<Lod> trials;ActionStats a,b;
        auto rank=[&](const ActionState& state,std::span<const ActionRecord> rows){std::vector<float> scores(rows.size());for(size_t i=0;i<rows.size();++i)if(ranking==NeuralRanking::ShortestEdge)scores[i]=-rows[i].x[59];else if(ranking==NeuralRanking::CurrentPlane)scores[i]=float(-state.teacher_cost(rows[i].action));return scores;};
        auto decision=[&](size_t n){return !reject||n+2>=m.view().triangles();};
        auto expected=execute_actions(m.view(),{},1,5,rank,[&](MeshView candidate){Lod l;l.data=copy_mesh(candidate);l.shared_vertices=false;trials.push_back(std::move(l));return decision(candidate.triangles());},&a,{},batch);
        gpu.reset();size_t at=0;auto actual=gpu.execute({},1,5,nullptr,ranking,741,batch,[&](DeviceMeshView candidate){require(at<trials.size(),"GPU emitted an extra trial");auto reference=trials[at++].data.view();require(candidate.faces==reference.triangles(),"GPU batch face count differs");
            auto cpu=evaluate(m.view(),reference,bounds(m.view()),e),device=audit.evaluate(m.view(),candidate,bounds(m.view()),e);require(cpu.complete==device.complete&&cpu.passed==device.passed&&std::abs(cpu.error-device.error)<1e-5&&cpu.changed_area==device.changed_area,"borrowed GPU audit differs");return decision(candidate.faces);},&b);
        require(at==trials.size()&&a.ranked==b.ranked&&a.trials==b.trials&&a.accepted==b.accepted&&a.rejected==b.rejected,"GPU executor work contract differs");require(actual.data.indices==expected.data.indices,"GPU executor output differs");}
    gpu.reset();ActionStats stopped;auto intact=gpu.execute({},1,5,nullptr,NeuralRanking::Constant,0,4,[](DeviceMeshView){return true;},&stopped,[]{return true;});require(!stopped.trials&&intact.data.indices==m.indices,"cancelled GPU executor changed source");
}
void placement_contracts(){auto m=plane(7),original=m;NeuralOptions options;options.memory_mib=256;GpuActionState gpu(m.view(),options,true);auto rows=gpu.placements({});
    auto found=std::find_if(rows.begin(),rows.end(),[](const auto& a){return a.action.from==24&&a.action.to==25;});require(found!=rows.end(),"free placement fixture has no interior action");auto action=found->action;
    auto alternatives=gpu.teacher_proposals(action);require(alternatives.size()==20,"teacher candidate contract");bool off_edge=false;for(auto& p:alternatives){for(float v:p.target)require(std::isfinite(v),"nonfinite teacher target");off_edge|=p.placement.position.z!=0;}require(off_edge,"teacher searches only the edge segment");
    Placement placement{{3.5f,3,.08f},{normalized({.15f,.2f,1}),{}}};DeviceMeshView candidate;require(gpu.trial(action,placement,candidate),"valid XYZ placement rejected");require(candidate.faces==m.view().triangles()-2,"placement removed wrong face count");
    auto prior=gpu.snapshot();auto bad=placement;bad.position.x=INFINITY;require(!gpu.trial(action,bad,candidate),"nonfinite placement accepted");require(same_mesh_data(prior.data.view(),gpu.snapshot().data.view()),"failed placement mutated incumbent");
    gpu.commit(action,placement);auto result=gpu.snapshot();require(!result.shared_vertices&&validate(result.data.view()).empty(),"placement output ownership/layout");bool moved=false;
    for(size_t i=0;i<result.data.positions.size();++i)if(result.data.positions[i].z!=0){moved=true;require(length(result.data.positions[i]-placement.position)<1e-6,"placement restricted to an endpoint");require(length(result.data.normals[i]-placement.normals[0])<1e-6,"learned normal was discarded");auto t=result.data.tangents[i];require(std::abs(dot(result.data.normals[i],{t.x,t.y,t.z}))<1e-6&&std::abs(length({t.x,t.y,t.z})-1)<1e-6,"tangent frame not transported");require(std::abs(result.data.uv[i].x-3.5f)<1e-6&&std::abs(result.data.uv[i].y-3)<1e-6,"attribute correspondence left original chart");}
    require(moved&&same_mesh_data(m.view(),original.view()),"placement did not move or changed source");auto committed_revision=gpu.view().revision;gpu.reset();require(same_mesh_data(gpu.snapshot().data.view(),prior.data.view()),"placement reset lost source data");require(gpu.view().revision>committed_revision,"reset reused a cached render revision");auto fresh=gpu.placements({});auto again=std::find_if(fresh.begin(),fresh.end(),[&](const auto& row){return row.action.from==action.from&&row.action.to==action.to;});require(again!=fresh.end(),"reset lost fixture action");placement.position.z=.04f;gpu.commit(again->action,placement);require(gpu.view().revision>committed_revision,"new trajectory reused a committed render revision");
    WeightsData weights;weights.architecture=placement_schema;weights.values.resize(placement_weight_count);ActionCuda policy(weights,options,7);std::vector<float> input(11*placement_features);auto prediction=policy.predict(input);require(prediction.size()==11*placement_outputs&&std::all_of(prediction.begin(),prediction.end(),[](float x){return x==0;}),"v3 native policy layout");
    GpuActionState reuse(m.view(),options);ActionStats stats;auto unchanged=reuse.execute({},m.view().triangles(),4,&policy,NeuralRanking::Learned,5,2,[](DeviceMeshView){return true;},&stats);require(unchanged.shared_vertices&&unchanged.data.positions.empty()&&unchanged.data.indices==m.indices,"v3 Reuse changed vertex ownership");
}
void coupled_placement(const Mesh& mesh){NeuralOptions options;options.memory_mib=256;GpuActionState gpu(mesh.view(),options,true);auto rows=gpu.placements({});
    auto found=std::find_if(rows.begin(),rows.end(),[](const auto& row){return row.action.from==7&&row.action.to==12;});require(found!=rows.end(),"seam action missing");
    Placement p{{2,1.5f,.02f},{normalized({.1f,0,1}),normalized({0,1,.1f})}};DeviceMeshView v;require(gpu.trial(found->action,p,v),"coupled seam placement rejected");gpu.commit(found->action,p);auto result=gpu.snapshot();unsigned left=0,right=0;
    for(size_t i=0;i<result.data.positions.size();++i)if(result.data.positions[i].z!=0){require(length(result.data.positions[i]-p.position)<1e-6,"seam cracked");if(result.data.uv[i].x<0){++right;require(length(result.data.normals[i]-p.normals[1])<1e-6,"second normal crossed seam");}else{++left;require(length(result.data.normals[i]-p.normals[0])<1e-6,"first normal crossed seam");}}
    require(left&&right&&validate(result.data.view()).empty(),"seam lost a wedge");
}
void incremental_audits(){auto m=plane(9);NeuralOptions options;options.memory_mib=256;GpuActionState state(m.view(),options,true);
    AuditCuda incremental(options,m.view());auto uncached=options;uncached.cache_rasters=false;AuditCuda full(uncached,m.view());
    EvalSettings e;e.profile=Profile::Attributes;e.screen_size=37;e.limit=4;e.views={3,2,719};e.supersample=e.max_supersample=3;
    for(unsigned step=0;step<3;++step){auto rows=state.teacher_actions({},2,137+step);require(!rows.empty(),"incremental raster fixture exhausted");
        for(auto& row:rows){auto proposals=state.teacher_proposals(row.action);unsigned valid=0;
            for(auto& proposal:proposals){DeviceMeshView candidate;if(!state.trial(row.action,proposal.placement,candidate))continue;++valid;
                auto a=incremental.evaluate(m.view(),candidate,bounds(m.view()),e),b=full.evaluate(m.view(),candidate,bounds(m.view()),e);
                require(a.complete==b.complete&&a.passed==b.passed&&a.error==b.error&&a.coverage==b.coverage&&a.changed_area==b.changed_area&&a.normal_degrees==b.normal_degrees&&a.views_evaluated==b.views_evaluated,"incremental raster changed exact audit");
                if(valid==1&&a.complete&&a.passed){double margin=std::max(a.error/e.limit,a.changed_area/e.max_changed_area);bool pruned=false;
                    auto tied=incremental.evaluate(m.view(),candidate,bounds(m.view()),e,nullptr,margin,&pruned);
                    require(pruned&&!tied.complete&&!tied.passed&&tied.views_evaluated<=a.views_evaluated,"incumbent tie was not pruned as an unknown bound");
                    auto better=incremental.evaluate(m.view(),candidate,bounds(m.view()),e,nullptr,std::nextafter(margin,INFINITY),&pruned);
                    require(!pruned&&better.complete&&better.error==a.error&&better.changed_area==a.changed_area,"incumbent pruning discarded an improving candidate");}
            }require(valid>1,"incremental raster needs distinct valid placements");
        }
        auto row=rows.front();bool committed=false;for(auto& p:state.teacher_proposals(row.action)){DeviceMeshView v;if(state.trial(row.action,p.placement,v)){state.commit(row.action,p.placement);committed=true;break;}}
        require(committed,"incremental raster commit fixture");
    }
}
void sparse_teacher(){auto m=plane(7);NeuralOptions options;options.memory_mib=256;GpuActionState state(m.view(),options,true);
    std::array<float,conditions> c{.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f};
    for(unsigned count:{1u,4u,16u}){auto sparse=state.teacher_actions(c,count,918),full=state.placements(c);
        for(auto& row:sparse){auto found=std::find_if(full.begin(),full.end(),[&](auto& x){return x.action==row.action;});require(found!=full.end()&&found->x==row.x,"sparse teacher changed selected features");state.teacher_proposals(row.action);}
        auto again=state.teacher_actions(c,count,918);require(again.size()==sparse.size(),"teacher selection count changed");
        for(size_t i=0;i<sparse.size();++i){require(again[i].action==sparse[i].action&&again[i].x==sparse[i].x,"teacher selection reused stale indices");auto proposals=state.teacher_proposals(again[i].action);require(proposals[0].placement.position.x==m.positions[again[i].action.from].x&&proposals[0].placement.position.y==m.positions[again[i].action.from].y,"teacher cached a different selected action");}
    }
}
void workspace_budget(){NeuralOptions options;options.memory_mib=128;MemoryScope memory(options);memory.budget.limit=1024;
    {gpu::Device first(options,true),second(options);{gpu::Buffer<std::byte> a(first,640);bool rejected=false;try{gpu::Buffer<std::byte> b(second,512);}catch(const gpu::ResourceError& e){rejected=e.kind==NeuralResourceLimit::WorkspaceMemory&&e.limit==1024;}require(rejected,"separate workspaces exceeded shared cap");}
        require(memory.budget.live==640,"idle pool capacity disappeared from budget");{gpu::Buffer<std::byte> reuse(first,512);require(memory.budget.live==640,"pool reuse double counted budget");}}
    require(memory.budget.live==0&&memory.budget.peak==640,"workspace allocation lifetime mismatch");
}
void strided_upload(){auto m=plane(5);struct Vertex {uint32_t prefix;Vec3 position;uint8_t gap[7];Vec3 normal;};std::vector<Vertex> vertices(m.positions.size());for(size_t i=0;i<vertices.size();++i){vertices[i].position=m.positions[i];vertices[i].normal=m.normals[i];}
    auto view=m.view();view.positions.data=reinterpret_cast<const std::byte*>(&vertices[0].position);view.positions.stride=sizeof(Vertex);view.normals.data=reinterpret_cast<const std::byte*>(&vertices[0].normal);view.normals.stride=sizeof(Vertex);
    NeuralOptions options;options.memory_mib=128;GpuActionState packed(m.view(),options),strided(view,options);auto a=packed.actions({}),b=strided.actions({});require(a.size()==b.size(),"strided upload topology");for(size_t i=0;i<a.size();++i)require(a[i].action==b[i].action&&a[i].x==b[i].x,"strided upload changed features");
}
int main(){try{if(!neural_available())return 77;
    for(unsigned n:{5u,7u}){auto m=plane(n);parity(m,8);m.normals.clear();m.uv.clear();m.colors.clear();m.tangents.clear();m.positions[n+1].z=.2f;parity(m,6);}
    auto seam=plane(5);std::vector<uint32_t> duplicate(25,UINT32_MAX);for(uint32_t y=0;y<5;++y){auto i=y*5+2;duplicate[i]=uint32_t(seam.positions.size());seam.positions.push_back(seam.positions[i]);seam.normals.push_back({0,1,0});seam.uv.push_back({-2,float(y)});seam.colors.push_back(seam.colors[i]);seam.tangents.push_back({0,0,1,-1});}
    for(uint32_t f=0;f<seam.view().triangles();++f){bool right=false;for(unsigned j=0;j<3;++j)right|=seam.positions[seam.indices[f*3+j]].x>2;seam.materials.push_back(uint16_t(right));if(right)for(unsigned j=0;j<3;++j){auto& i=seam.indices[f*3+j];if(duplicate[i]!=UINT32_MAX)i=duplicate[i];}}seam.double_sided={1,1};parity(seam,8);
    Mesh tetra;tetra.positions={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};tetra.indices={0,2,1,0,1,3,0,3,2,1,2,3};parity(tetra,1);
    auto unsafe=plane(5);unsafe.indices.insert(unsafe.indices.end(),{6,7,12,6,7,17,1,1,2});parity(unsafe,3);
    Mesh bow;bow.positions={{0,0,0},{1,0,0},{0,1,0},{-1,0,0},{0,-1,0}};bow.indices={0,1,2,0,3,4};parity(bow,1);
    executor();placement_contracts();coupled_placement(seam);incremental_audits();sparse_teacher();workspace_budget();strided_upload();std::cout<<"GPU action contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
