#include "neural_action_data.hpp"
#include <random>
#include <numeric>
#include <iostream>
#include <csignal>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
int main(int argc,char** argv){try{
    if(argc<3)throw std::invalid_argument("blitz-neural-action-prepare ASSET OUTPUT [--states N] [--pixels N] [--previous-steps N] [--source-limit N] [--adjacent-limit N] [--minutes N] [--model FILE] [--gpu-memory-mib N]");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);uint32_t states=64,previous_steps=0;uint64_t memory_mib=6144;double pixels=32,minutes=5,source_limit=3,adjacent_limit=3;fs::path model;
    auto count=[](const char* text){std::string value=text;if(value.empty()||value.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid state count");auto n=std::stoull(value);if(n>4096)throw std::invalid_argument("state count exceeds 4096");return uint32_t(n);};
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing option");std::string k=argv[i];if(k=="--states")states=count(argv[i+1]);else if(k=="--pixels")pixels=std::stod(argv[i+1]);else if(k=="--minutes")minutes=std::stod(argv[i+1]);else if(k=="--model")model=argv[i+1];else if(k=="--previous-steps")previous_steps=count(argv[i+1]);else if(k=="--source-limit")source_limit=std::stod(argv[i+1]);else if(k=="--adjacent-limit")adjacent_limit=std::stod(argv[i+1]);else if(k=="--gpu-memory-mib")memory_mib=std::stoull(argv[i+1]);else throw std::invalid_argument("unknown action preparation option");}
    if(!states||states>4096||previous_steps>4096||memory_mib<128||memory_mib>65536||!std::isfinite(minutes)||minutes<=0||minutes>50||!std::isfinite(pixels)||pixels<16||pixels>512||!std::isfinite(source_limit)||source_limit<=0||source_limit>16||!std::isfinite(adjacent_limit)||adjacent_limit<=0||adjacent_limit>16)throw std::invalid_argument("invalid bounded preparation settings");
    NeuralOptions options;options.memory_mib=uint32_t(memory_mib);NeuralStats audit_stats;
    fs::path output=argv[2];if(fs::exists(output/"index.json"))throw std::invalid_argument("choose a fresh action dataset directory");fs::create_directories(output);
    auto [mesh,metadata]=training_mesh(argv[1]);auto source=mesh.view();const auto bounds=blitz::bounds(source);
    auto start=std::chrono::steady_clock::now();auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto cancel=[&]{return bool(stopped)||seconds()>minutes*60;};
    auto e=action_eval(previous_steps?std::min(512.,pixels*2):pixels,source_limit);e.cancelled=cancel;auto adjacent=e;adjacent.limit=adjacent_limit;
    ActionState state(source);ActionData data;std::mt19937 random(0xB1172026);Lod previous;MeshView previous_view=source;bool emitted=false;
    std::unique_ptr<ActionCuda> policy;if(!model.empty())policy=std::make_unique<ActionCuda>(load_weights(model),options);
    json contract={{"schema",action_schema},{"asset",metadata},{"source_manifest_sha256",file_sha256("research/corpus.json")},{"training_selection_sha256",file_sha256("research/neural/training-manifest.json")},{"protocol_sha256",file_sha256("research/PROTOCOL.md")},{"binary_sha256",file_sha256("/proc/self/exe")},{"states_requested",states},{"pixels",pixels},{"source_limit",e.limit},{"adjacent_limit",e.limit},{"area_limit",e.max_changed_area},{"views",{6,2}},{"view_seed",e.views.rotation_seed},{"supersample",4},{"max_supersample",8},{"pool",{8,4,4}},{"previous_emitted_lod","source (single scheduled transition)"},{"policy_sha256",model.empty()?"":file_sha256(model)}};
    contract["adjacent_limit"]=adjacent_limit;contract["previous_steps"]=previous_steps;contract["previous_pixels"]=e.screen_size;contract["previous_emitted_lod"]=previous_steps?"audited fixed intermediate LOD":"source (single scheduled transition)";
    const std::vector<double> fractions=previous_steps?std::vector<double>{.02,.1,.5,.9}:std::vector<double>{.1};contract["target_fractions"]=fractions;
    contract["gpu_memory_mib"]=memory_mib;
    write_json(output/"contract.json",contract);json trace=json::array();bool exhausted=false;uint64_t queries=0;uint32_t accepted=0;
    for(uint32_t step=0;step<states+previous_steps&&!cancel();++step){
        if(previous_steps&&step==previous_steps){auto a=evaluate(source,state.view(),bounds,e),b=evaluate(previous_view,state.view(),bounds,adjacent);
            if(!a.complete||!a.passed||!b.complete||!b.passed||state.view().triangles()>=source.triangles())throw std::runtime_error("preceding LOD failed CPU confirmation or did not reduce");previous=state.lod();previous_view=previous.view(source);emitted=true;e.screen_size=pixels;adjacent.screen_size=pixels;}
        auto fraction=fractions[step%fractions.size()];auto rows=state.actions(condition(e,adjacent.limit,fraction));if(rows.empty()){exhausted=true;break;}
        std::vector<uint32_t> teacher(rows.size()),shuffled(rows.size()),selected;std::iota(teacher.begin(),teacher.end(),0);shuffled=teacher;std::shuffle(shuffled.begin(),shuffled.end(),random);
        std::vector<double> costs(rows.size());for(uint32_t i=0;i<rows.size();++i)costs[i]=state.teacher_cost(rows[i].action);
        std::stable_sort(teacher.begin(),teacher.end(),[&](auto a,auto b){return costs[a]<costs[b];});
        auto add=[&](const auto& list,uint32_t count){for(auto id:list){if(std::find(selected.begin(),selected.end(),id)!=selected.end())continue;selected.push_back(id);if(!--count)break;}};
        add(teacher,8);add(shuffled,4);
        std::vector<float> logits;
        if(policy){std::vector<float> x;for(auto& row:rows)x.insert(x.end(),row.x.begin(),row.x.end());logits=policy->predict(x);auto order=teacher;std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return logits[a*3]>logits[b*3];});add(order,4);}else add(shuffled,4);
        auto first=data.labels.size();json state_trace={{"revision",step},{"triangles",state.view().triangles()},{"previous_triangles",previous_view.triangles()},{"pixels",e.screen_size},{"target_fraction",fraction},{"legal_actions",rows.size()},{"queries",json::array()}};
        std::vector<uint32_t> safe;double best=INFINITY;
        for(auto i:selected){auto candidate=state.trial(rows[i].action);auto a=evaluate_cuda(source,candidate.view(source),bounds,e,options,&audit_stats),b=evaluate_cuda(previous_view,candidate.view(source),bounds,adjacent,options,&audit_stats);++queries;
            if(!action_audit_known(a,e)||!action_audit_known(b,adjacent)){auto& f=audit_stats.first_resource_failure;write_json(output/"incomplete.json",{{"state",state_trace},{"queries",queries},{"cancelled",cancel()},{"resource_limited",a.resource_limited||b.resource_limited},{"gpu_peak_bytes",audit_stats.gpu_peak_bytes},{"resource_kind",int(f.kind)},{"requested_bytes",f.requested},{"limit_bytes",f.limit}});throw std::runtime_error("action audit has no verdict; no label or score assigned");}
            auto label=uint8_t(Queried|(a.passed?SourcePass:0)|(b.passed?AdjacentPass:0));if(a.passed&&b.passed){safe.push_back(i);best=std::min(best,costs[i]);}
            data.x.insert(data.x.end(),rows[i].x.begin(),rows[i].x.end());data.labels.push_back(label);data.from.push_back(rows[i].action.from);data.to.push_back(rows[i].action.to);
            state_trace["queries"].push_back({{"from",rows[i].action.from},{"to",rows[i].action.to},{"source_pass",a.passed},{"adjacent_pass",b.passed},{"source_error",a.error},{"adjacent_error",b.error},{"area",a.changed_area},{"teacher_cost",costs[i]}});}
        std::vector<uint32_t> preferred;for(size_t j=0;j<selected.size();++j)if((data.labels[first+j]&3)==3&&costs[selected[j]]<=best*1.25+1e-10){data.labels[first+j]|=Preferred;preferred.push_back(selected[j]);state_trace["queries"][j]["preferred"]=true;}
        data.offsets.push_back(uint32_t(data.labels.size()));data.progress.push_back(float(1-double(state.view().triangles())/source.triangles()));
        if(preferred.empty()){exhausted=true;state_trace["stopped"]="no safe preferred queried action";trace.push_back(state_trace);break;}
        auto chosen=preferred[random()%preferred.size()];if(policy)chosen=*std::max_element(safe.begin(),safe.end(),[&](auto a,auto b){return logits[a*3]<logits[b*3];});
        state_trace["selected"]={{"from",rows[chosen].action.from},{"to",rows[chosen].action.to}};state.commit(rows[chosen].action);++accepted;trace.push_back(state_trace);
        std::cout<<json({{"state",step},{"queries",queries},{"triangles",state.view().triangles()},{"seconds",seconds()}}).dump()<<std::endl;
    }
    bool complete=!cancel()&&(data.states()==states+previous_steps||exhausted)&&(!previous_steps||emitted);save_actions(output/"actions.bin",data);
    auto final=evaluate(source,state.view(),bounds,e),adjacent_final=evaluate(previous_view,state.view(),bounds,adjacent);complete&=final.complete&&final.passed&&adjacent_final.complete&&adjacent_final.passed;
    write_json(output/"trajectory.json",trace);write_json(output/"index.json",{{"schema",action_schema},{"complete",complete},{"asset",argv[1]},{"category",metadata.at("category")},{"contract_sha256",file_sha256(output/"contract.json")},{"path","actions.bin"},{"sha256",file_sha256(output/"actions.bin")},{"states",data.states()},{"queries",queries},{"accepted",accepted},{"source_triangles",source.triangles()},{"teacher_triangles",state.view().triangles()},{"reference_confirmed",final.complete&&final.passed&&adjacent_final.complete&&adjacent_final.passed},{"preceding_lod_emitted",emitted},{"previous_triangles",previous_view.triangles()},{"seconds",seconds()},{"training_started",false}});
    return complete?0:2;
}catch(const std::exception& e){std::cerr<<"action preparation: "<<e.what()<<'\n';return 1;}}
