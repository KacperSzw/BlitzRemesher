#include "neural_action_data.hpp"
#include "neural_action_gpu.hpp"
#include "neural_memory.hpp"
#include "neural_json.hpp"
#include <csignal>
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
int main(int argc,char** argv){try{
    if(argc<3)throw std::invalid_argument("blitz-neural-placement-prepare ASSET OUTPUT [--states N] [--pool N] [--pixels N] [--previous-steps N] [--source-limit N] [--adjacent-limit N] [--minutes N] [--gpu-memory-mib N] [--model MODEL] [--seed N]");
    uint32_t states=32,pool=16,previous_steps=0,seed=101;double pixels=128,minutes=5,source_limit=3,adjacent_limit=3;NeuralOptions options;fs::path model;
    auto integer=[](const char* value){std::string s=value;if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid integer option");auto x=std::stoull(s);if(x>65536)throw std::invalid_argument("integer option too large");return uint32_t(x);};
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing preparation option");std::string k=argv[i];if(k=="--states")states=integer(argv[i+1]);else if(k=="--seed")seed=integer(argv[i+1]);else if(k=="--pool")pool=integer(argv[i+1]);else if(k=="--previous-steps")previous_steps=integer(argv[i+1]);else if(k=="--pixels")pixels=std::stod(argv[i+1]);else if(k=="--source-limit")source_limit=std::stod(argv[i+1]);else if(k=="--adjacent-limit")adjacent_limit=std::stod(argv[i+1]);else if(k=="--model")model=argv[i+1];else if(k=="--minutes")minutes=std::stod(argv[i+1]);else if(k=="--gpu-memory-mib")options.memory_mib=integer(argv[i+1]);else throw std::invalid_argument("unknown preparation option");}
    if(!states||states>4096||previous_steps>4096||!pool||pool>16||!std::isfinite(pixels)||pixels<16||pixels>512||!std::isfinite(minutes)||minutes<=0||minutes>50||!std::isfinite(source_limit)||source_limit<=0||source_limit>16||!std::isfinite(adjacent_limit)||adjacent_limit<=0||adjacent_limit>16)throw std::invalid_argument("preparation bounds");
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);fs::path output=argv[2];if(fs::exists(output/"contract.json"))throw std::invalid_argument("choose a fresh placement dataset directory");fs::create_directories(output);
    MemoryScope memory(options);auto [mesh,metadata]=training_mesh(argv[1]);auto source=mesh.view();auto bounds=blitz::bounds(source);GpuActionState state(source,options,true);AuditCuda audit(options,source);NeuralStats stats;ActionData data;data.architecture=placement_schema;std::unique_ptr<ActionCuda> policy;if(!model.empty()){auto w=load_weights(model);if(w.architecture!=placement_schema)throw std::invalid_argument("placement teacher requires a v3 policy");policy=std::make_unique<ActionCuda>(w,options);}
    auto start=std::chrono::steady_clock::now();auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto cancel=[&]{return bool(stopped)||seconds()>minutes*60;};
    auto e=action_eval(previous_steps?std::min(512.,pixels*2):pixels,source_limit);e.cancelled=cancel;auto adjacent=e;adjacent.limit=adjacent_limit;Lod previous;MeshView previous_view=source;bool emitted=false,exhausted=false,unknown=false;uint64_t queries=0,reused_queries=0,reused_adjacent=0,pruned_queries=0;uint32_t accepted=0;
    json contract={{"schema",placement_schema},{"teacher_version",1},{"seed",seed},{"asset",metadata},{"source_manifest_sha256",file_sha256("research/corpus.json")},{"training_selection_sha256",file_sha256("research/neural/training-manifest.json")},{"protocol_sha256",file_sha256("research/PROTOCOL.md")},{"binary_sha256",file_sha256("/proc/self/exe")},{"states_requested",states},{"pool",pool},{"pixels",pixels},{"previous_steps",previous_steps},{"source_limit",source_limit},{"adjacent_limit",adjacent_limit},{"area_limit",e.max_changed_area},{"views",{6,2}},{"view_seed",e.views.rotation_seed},{"supersample",4},{"max_supersample",8},{"gpu_memory_mib",options.memory_mib},{"teacher_candidates",policy?21:20},{"policy_sha256",model.empty()?"":file_sha256(model)},{"target","one audited joint XYZ/normal candidate per edge; no averaging"},{"preference","minimum triangles, then minimum worst normalized source/adjacent error"}};
    write_json(output/"contract.json",contract);json trace=json::array();
    for(uint32_t step=0;step<states+previous_steps&&!cancel();++step){
        if(previous_steps&&step==previous_steps){previous=state.snapshot();auto a=audit.evaluate(source,previous.data.view(),bounds,e,&stats);if(!a.complete||!a.passed||previous.data.view().triangles()>=source.triangles())throw std::runtime_error("preceding LOD failed its fixed audit");previous_view=previous.data.view();emitted=true;e.screen_size=pixels;adjacent.screen_size=pixels;}
        double fraction=std::array{.1,.5,.9,.02}[step%4];auto rows=state.teacher_actions(condition(e,adjacent.limit,fraction),pool,seed,policy.get());if(rows.empty()){exhausted=true;break;}
        size_t first=data.labels.size(),chosen=SIZE_MAX;uint32_t best_faces=UINT32_MAX;double best_margin=INFINITY;std::vector<GpuActionState::Proposal> winners;std::vector<Action> actions;
        json state_trace={{"revision",step},{"triangles",state.view().faces},{"previous_triangles",previous_view.triangles()},{"pixels",e.screen_size},{"queries",json::array()}};
        for(auto& row:rows){bool found=false;uint8_t best_label=0;double best=INFINITY;uint32_t faces=UINT32_MAX;GpuActionState::Proposal winner{};
            auto alternatives=state.teacher_proposals(row.action);
            struct Query {Measurement source,adjacent;uint32_t faces{};bool valid{},pruned{};};std::array<Query,21> evaluated{};
            static_assert(sizeof(Placement)==9*sizeof(float));
            for(size_t index=0;index<alternatives.size()&&!cancel();++index){auto& proposal=alternatives[index];auto& q=evaluated[index];size_t prior=0;
                while(prior<index&&std::memcmp(&proposal.placement,&alternatives[prior].placement,sizeof(Placement)))++prior;
                if(prior<index){q=evaluated[prior];++reused_queries;}
                else {DeviceMeshView candidate;q.valid=state.trial(row.action,proposal.placement,candidate);if(q.valid){q.faces=candidate.faces;
                    double cutoff=found&&(best_label&27)==27?best:INFINITY;
                    q.source=audit.evaluate(source,candidate,bounds,e,&stats,cutoff,std::isfinite(cutoff)?&q.pruned:nullptr);
                    if(!q.pruned){if(!emitted&&e.limit==adjacent.limit){q.adjacent=q.source;++reused_adjacent;}else q.adjacent=audit.evaluate(previous_view,candidate,bounds,adjacent,&stats,cutoff,std::isfinite(cutoff)?&q.pruned:nullptr);}}}
                if(!q.valid)continue;auto& a=q.source;auto& b=q.adjacent;++queries;
                if(q.pruned){++pruned_queries;state_trace["queries"].push_back({{"from",row.action.from},{"to",row.action.to},{"candidate",index},{"known_mask",0},{"pruned_by_incumbent",true},{"faces",q.faces}});continue;}
                bool ka=action_audit_known(a,e),kb=action_audit_known(b,adjacent);uint8_t label=uint8_t((ka?SourceKnown:0)|(kb?AdjacentKnown:0)|(ka&&a.passed?PlacementSourcePass:0)|(kb&&b.passed?PlacementAdjacentPass:0));bool safe=(label&27)==27;
                double margin=std::max({a.error/e.limit,b.error/adjacent.limit,a.changed_area/e.max_changed_area,b.changed_area/adjacent.max_changed_area});if(!std::isfinite(margin))margin=INFINITY;
                state_trace["queries"].push_back({{"from",row.action.from},{"to",row.action.to},{"candidate",index},{"known_mask",label},{"source_error",a.error},{"adjacent_error",b.error},{"faces",q.faces}});
                if(!found||(safe&&((best_label&3)!=3))||(safe==((best_label&3)==3)&&margin<best)){found=true;winner=proposal;best_label=label;best=margin;faces=q.faces;}
                if(!ka||!kb){unknown=true;break;}
            }
            if(found){bool safe=(best_label&27)==27;if(safe)best_label|=uint8_t(PositionKnown|(winner.normal_mask&1?Normal0Known:0)|(winner.normal_mask&2?Normal1Known:0));
                data.x.insert(data.x.end(),row.x.begin(),row.x.end());data.labels.push_back(best_label);data.from.push_back(row.action.from);data.to.push_back(row.action.to);
                for(unsigned j=0;j<9;++j)data.targets.push_back((best_label&(32u<<(j/3)))?winner.target[j]:0);
                actions.push_back(row.action);winners.push_back(winner);if(safe&&(faces<best_faces||(faces==best_faces&&best<best_margin))){chosen=actions.size()-1;best_faces=faces;best_margin=best;}}
            if(unknown||cancel())break;
        }
        if(data.labels.size()>first){data.offsets.push_back(uint32_t(data.labels.size()));data.progress.push_back(float(1-double(state.view().faces)/source.triangles()));}
        if(unknown||cancel()){state_trace["complete"]=false;trace.push_back(state_trace);break;}
        if(chosen==SIZE_MAX){exhausted=true;state_trace["stopped"]="no feasible queried placement";trace.push_back(state_trace);break;}
        data.labels[first+chosen]|=PlacementPreferred;state.commit(actions[chosen],winners[chosen].placement);++accepted;state_trace["selected"]={{"from",actions[chosen].from},{"to",actions[chosen].to},{"triangles",state.view().faces}};trace.push_back(state_trace);
        std::cout<<json({{"state",step},{"queries",queries},{"triangles",state.view().faces},{"seconds",seconds()}}).dump()<<std::endl;
    }
    auto final=state.snapshot();auto a=audit.evaluate(source,final.data.view(),bounds,e,&stats),b=!emitted&&e.limit==adjacent.limit?a:audit.evaluate(previous_view,final.data.view(),bounds,adjacent,&stats);
    bool complete=!cancel()&&!unknown&&(data.states()==states+previous_steps||exhausted)&&(!previous_steps||emitted)&&a.complete&&a.passed&&b.complete&&b.passed;
    save_actions(output/"actions.bin",data);write_json(output/"trajectory.json",trace);write_json(output/"reuse.json",{{"duplicate_proposals",reused_queries},{"identical_adjacent_audits",reused_adjacent},{"pruned_candidates",pruned_queries}});
    write_json(output/"index.json",{{"schema",placement_schema},{"complete",complete},{"asset",argv[1]},{"category",metadata.at("category")},{"contract_sha256",file_sha256(output/"contract.json")},{"path","actions.bin"},{"sha256",file_sha256(output/"actions.bin")},{"states",data.states()},{"queries",queries},{"accepted",accepted},{"source_triangles",source.triangles()},{"teacher_triangles",final.data.view().triangles()},{"reference_confirmed",a.complete&&a.passed&&b.complete&&b.passed},{"preceding_lod_emitted",emitted},{"previous_triangles",previous_view.triangles()},{"seconds",seconds()},{"training_started",false},{"audit",neural_json(stats)}});
    return complete?0:2;
}catch(const std::exception& e){std::cerr<<"placement preparation: "<<e.what()<<'\n';return 1;}}
