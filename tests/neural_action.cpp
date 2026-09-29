#include "neural_action.hpp"
#include "neural_action_cache.hpp"
#include "chain_hooks.hpp"
#include <iostream>
#include <numeric>
using namespace blitz;using namespace blitz::neural;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
Mesh plane(unsigned n) {
    Mesh m;for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){m.positions.push_back({float(x),float(y),0});m.uv.push_back({float(x),float(y)});m.normals.push_back({0,0,1});}
    for(unsigned y=0;y+1<n;++y)for(unsigned x=0;x+1<n;++x){uint32_t a=y*n+x;m.indices.insert(m.indices.end(),{a,a+1,a+n,a+1,a+n+1,a+n});}
    m.double_sided={1};return m;
}
int main(){try{
    auto cache_mesh=plane(5);cache_mesh.materials.resize(cache_mesh.view().triangles(),0);
    for(auto bytes:{cache_mesh.indices.size()*sizeof(uint32_t),cache_mesh.indices.size()*sizeof(uint32_t)*2}){
        ActionRejections cache(bytes);EvalSettings e;cache.configure(e);cache.insert(cache_mesh.view());
        check(cache.contains(cache_mesh.view())==(bytes>=cache_mesh.indices.size()*sizeof(uint32_t)+cache_mesh.materials.size()*sizeof(uint16_t)),"cache payload bound changed verdict");
    }
    ActionRejections cache;EvalSettings cache_settings;cache.configure(cache_settings);cache.insert(cache_mesh.view());check(cache.contains(cache_mesh.view()),"identical rejection was not reused");
    auto altered=cache_mesh;altered.materials[0]=1;check(!cache.contains(altered.view()),"cache ignored material identity");altered=cache_mesh;std::swap(altered.indices[0],altered.indices[1]);check(!cache.contains(altered.view()),"cache ignored index identity");
    for(const auto& change:std::array<std::function<void(EvalSettings&)>,4>{[](auto& e){e.screen_size=64;},[](auto& e){e.limit=5;},[](auto& e){e.weights.normal=0;},[](auto& e){++e.views.rotation_seed;}}){
        cache.configure(cache_settings);cache.insert(cache_mesh.view());auto next=cache_settings;change(next);cache.configure(next);check(!cache.contains(cache_mesh.view()),"cache reused a different audit contract");}
    for(double limit:{1.5,3.5}){EvalSettings e;e.limit=limit;e.max_changed_area=.25;Measurement m;
        check(action_audit_known(m,e),"complete pass lacks a label");m.complete=false;m.passed=false;
        check(!action_audit_known(m,e),"cancelled audit received a negative label");m.views_evaluated=3;m.error=limit;
        check(!action_audit_known(m,e),"partial passing views received a negative label");m.error=limit+.5;
        check(action_audit_known(m,e),"witnessed pixel failure was discarded");m.error=limit;m.changed_area=.5;
        check(action_audit_known(m,e),"witnessed area failure was discarded");m.resource_limited=true;
        check(!action_audit_known(m,e),"resource failure received a label");m.resource_limited=false;m.error=NAN;
        check(!action_audit_known(m,e),"NaN audit received a label");}
    for(unsigned n:{5u,7u}){auto m=plane(n);auto source=copy_mesh(m.view());ActionState state(m.view());auto actions=state.actions({});check(!actions.empty(),"plane lacks legal actions");
        auto action=actions[actions.size()/2].action;auto before=state.lod().data.indices;auto trial=state.trial(action);
        check(state.lod().data.indices==before,"trial mutated current mesh");check(trial.shared_vertices&&trial.data.positions.empty(),"trial copied borrowed vertices");
        check(trial.view(m.view()).triangles()<m.view().triangles(),"legal action did not reduce");check(uv_distortion(trial.view(m.view())).negative_uv_faces==0,"action flipped UVs");
        state.commit(action);check(state.lod().data.indices==trial.data.indices,"commit differs from trial");check(!state.legal(action),"stale action remained legal");
        bool stale=false;try{state.trial(action);}catch(const std::invalid_argument&){stale=true;}check(stale,"stale action executed");
        check(same_mesh_data(m.view(),source.view()),"action modified source streams");
        uint32_t calls=0;ActionStats stats;auto rank=[&](const ActionState&,std::span<const ActionRecord> rows){++calls;return std::vector<float>(rows.size(),0);};
        auto reduced=execute_actions(m.view(),{},1,4,rank,[](MeshView){return true;},&stats);
        check(stats.accepted==4&&calls==4,"executor reused scores after changing topology");check(reduced.view(m.view()).triangles()<m.view().triangles(),"executor did not reduce");
        auto rejected=execute_actions(m.view(),{},1,3,rank,[](MeshView){return false;},&stats);
        check(stats.trials==3&&stats.accepted==0&&same_mesh_data(rejected.view(m.view()),m.view()),"rejected transaction changed incumbent or work budget");
        auto cancelled=execute_actions(m.view(),{},1,5,rank,[](MeshView){return true;},&stats,[]{return true;});
        check(stats.trials==0&&same_mesh_data(cancelled.view(m.view()),m.view()),"cancellation changed source");
        bool nonfinite=false;try{execute_actions(m.view(),{},1,2,[](const ActionState&,std::span<const ActionRecord> rows){return std::vector<float>(rows.size(),NAN);},[](MeshView){return true;});}catch(const std::invalid_argument&){nonfinite=true;}
        check(nonfinite,"nonfinite action scores accepted");
    }
    auto seam=plane(5);std::vector<uint32_t> duplicate(25,UINT32_MAX);
    for(uint32_t batch_size:{2u,4u}){auto m=plane(9);ActionState s(m.view());auto rows=s.actions({});std::vector<uint32_t> order(rows.size());std::iota(order.begin(),order.end(),0);
        auto batch=s.independent(rows,order,batch_size,m.view().triangles()-1);check(batch.size()>1,"independent batch lacks parallel actions");auto trial=s.trial(batch);auto source=copy_mesh(m.view());
        check(same_mesh_data(s.view(),source.view()),"batch trial mutated incumbent");s.commit(batch);check(s.lod().data.indices==trial.data.indices,"batch commit differs from audited trial");
        bool stale=false;try{s.trial(batch);}catch(const std::invalid_argument&){stale=true;}check(stale,"stale batch executed");
        ActionStats stats;uint32_t calls=0;auto rank=[&](const ActionState&,std::span<const ActionRecord> a){++calls;return std::vector<float>(a.size(),0);};
        auto combined=execute_actions(m.view(),{},1,3,rank,[](MeshView){return true;},&stats,{},batch_size);
        check(stats.accepted>stats.trials&&calls==stats.trials,"batch did not share audits or re-rank after commit");check(combined.view(m.view()).triangles()<m.view().triangles(),"batch failed to reduce");
        auto limited=execute_actions(m.view(),{},m.view().triangles()-3,10,rank,[](MeshView){return true;},&stats,{},batch_size);
        check(limited.view(m.view()).triangles()>=m.view().triangles()-3,"batch undershot triangle target");
        auto backed=execute_actions(m.view(),{},1,8,rank,[&](MeshView candidate){return candidate.triangles()+2>=m.view().triangles();},&stats,{},batch_size);
        check(stats.accepted>0&&stats.rejected>0&&backed.view(m.view()).triangles()+2>=m.view().triangles(),"batch backoff lost the last audited incumbent");
        ActionState overlap(m.view());std::array<Action,2> duplicate_actions{rows[0].action,rows[0].action};bool rejected=false;try{overlap.trial(duplicate_actions);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"overlapping batch footprints accepted");
    }
    for(uint32_t y=0;y<5;++y){auto i=y*5+2;duplicate[i]=uint32_t(seam.positions.size());seam.positions.push_back(seam.positions[i]);seam.normals.push_back({0,1,0});seam.uv.push_back({-2,float(y)});}
    for(uint32_t f=0;f<seam.view().triangles();++f){bool right=false;for(unsigned j=0;j<3;++j)right|=seam.positions[seam.indices[f*3+j]].x>2;
        seam.materials.push_back(uint16_t(right));if(right)for(unsigned j=0;j<3;++j){auto& i=seam.indices[f*3+j];if(duplicate[i]!=UINT32_MAX)i=duplicate[i];}}
    seam.double_sided={1,1};ActionState charts(seam.view());auto chart_actions=charts.actions({});
    auto found=std::find_if(chart_actions.begin(),chart_actions.end(),[](auto& row){return row.action.from==12&&row.action.to==17;});
    check(found!=chart_actions.end(),"unambiguous coupled seam locked");auto collapsed=charts.trial(found->action);auto out=collapsed.view(seam.view());
    check(std::find(out.indices.begin(),out.indices.end(),12)==out.indices.end()&&std::find(out.indices.begin(),out.indices.end(),duplicate[12])==out.indices.end(),"seam collapsed on one chart only");
    check(std::find(out.indices.begin(),out.indices.end(),17)!=out.indices.end()&&std::find(out.indices.begin(),out.indices.end(),duplicate[17])!=out.indices.end(),"seam destination attributes merged");
    for(uint32_t f=0;f<out.triangles();++f)for(unsigned j=0;j<3;++j){auto i=out.indices[f*3+j];if(out.positions[i].x==2)check(out.material(f)==uint16_t(i>=25),"wedge moved across material chart");}
    Mesh tetra;tetra.positions={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};tetra.indices={0,2,1,0,1,3,0,3,2,1,2,3};
    check(ActionState(tetra.view()).actions({}).empty(),"tetrahedron creates duplicate faces");
    auto nonmanifold=plane(5);nonmanifold.indices.insert(nonmanifold.indices.end(),{6,7,12,6,7,17});ActionState unsafe(nonmanifold.view());
    check(!unsafe.legal({6,7,0}),"nonmanifold edge unlocked");
    Mesh bow;bow.positions={{0,0,0},{1,0,0},{0,1,0},{-1,0,0},{0,-1,0}};bow.indices={0,1,2,0,3,4};ActionState bow_state(bow.view());
    for(const auto& row:bow_state.actions({}))check(row.action.from!=0&&row.action.to!=0,"disconnected coincident fan unlocked");
    auto chain_mesh=plane(7);Settings settings;settings.levels=3;settings.base_pixels=32;settings.last_pixels=16;settings.candidate_budget=2;settings.beam_width=1;settings.research.output=OutputMode::Reuse;settings.research.chain=ChainMode::Direct;
    bool saw_previous=false;detail::GenerationHooks hooks;
    hooks.propose_guarded=[&](MeshView input,MeshView fixed,MeshView previous,const Bounds&,const ReduceSettings& rs,const EvalSettings& a,const EvalSettings& b){
        check(same_mesh_data(fixed,chain_mesh.view()),"guarded source drifted");check(same_mesh_data(input,fixed),"direct proposal changed origin");
        saw_previous|=previous.triangles()<fixed.triangles();check(a.screen_size==b.screen_size,"source/adjacent cameras differ");
        ActionStats stats;return execute_actions(input,{},rs.target_triangles,4,[](const ActionState&,std::span<const ActionRecord> rows){return std::vector<float>(rows.size(),0);},[](MeshView){return true;},&stats);};
    hooks.evaluate=[](MeshView,MeshView,const Bounds&,const EvalSettings&){return Measurement{};};hooks.confirm=[](Result&){return true;};
    auto chain=detail::generate_with_hooks(chain_mesh.view(),settings,{},&hooks);check(chain.lods.size()==3&&saw_previous,"guarded hook lost preceding emitted LOD");
    std::cout<<"action contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
