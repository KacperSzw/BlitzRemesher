#include "neural_action_data.hpp"
#include <random>
#include <numeric>
#include <iostream>
#include <csignal>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
int main(int argc,char** argv){try{
    if(argc<3)throw std::invalid_argument("blitz-neural-action-prepare ASSET OUTPUT [--states N] [--pixels N] [--minutes N] [--model FILE]");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);uint32_t states=64;double pixels=32,minutes=5;fs::path model;
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing option");std::string k=argv[i];if(k=="--states")states=uint32_t(std::stoul(argv[i+1]));else if(k=="--pixels")pixels=std::stod(argv[i+1]);else if(k=="--minutes")minutes=std::stod(argv[i+1]);else if(k=="--model")model=argv[i+1];else throw std::invalid_argument("unknown action preparation option");}
    if(!states||states>4096||!std::isfinite(minutes)||minutes<=0||minutes>50||!std::isfinite(pixels)||pixels<16||pixels>512)throw std::invalid_argument("invalid bounded preparation settings");
    fs::path output=argv[2];if(fs::exists(output/"index.json"))throw std::invalid_argument("choose a fresh action dataset directory");fs::create_directories(output);
    auto [mesh,metadata]=training_mesh(argv[1]);auto source=mesh.view();const auto bounds=blitz::bounds(source);
    auto start=std::chrono::steady_clock::now();auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto cancel=[&]{return bool(stopped)||seconds()>minutes*60;};
    auto e=action_eval(pixels);e.cancelled=cancel;ActionState state(source);ActionData data;std::mt19937 random(0xB1172026);
    std::unique_ptr<ActionCuda> policy;if(!model.empty())policy=std::make_unique<ActionCuda>(load_weights(model),NeuralOptions{});
    json contract={{"schema",action_schema},{"asset",metadata},{"source_manifest_sha256",file_sha256("research/corpus.json")},{"training_selection_sha256",file_sha256("research/neural/training-manifest.json")},{"protocol_sha256",file_sha256("research/PROTOCOL.md")},{"binary_sha256",file_sha256("/proc/self/exe")},{"states_requested",states},{"pixels",pixels},{"source_limit",e.limit},{"adjacent_limit",e.limit},{"area_limit",e.max_changed_area},{"views",{6,2}},{"view_seed",e.views.rotation_seed},{"supersample",4},{"max_supersample",8},{"pool",{8,4,4}},{"previous_emitted_lod","source (single scheduled transition)"},{"policy_sha256",model.empty()?"":file_sha256(model)}};
    write_json(output/"contract.json",contract);json trace=json::array();bool exhausted=false;uint64_t queries=0;uint32_t accepted=0;
    for(uint32_t step=0;step<states&&!cancel();++step){auto rows=state.actions(condition(e,e.limit,.1));if(rows.empty()){exhausted=true;break;}
        std::vector<uint32_t> teacher(rows.size()),shuffled(rows.size()),selected;std::iota(teacher.begin(),teacher.end(),0);shuffled=teacher;std::shuffle(shuffled.begin(),shuffled.end(),random);
        std::vector<double> costs(rows.size());for(uint32_t i=0;i<rows.size();++i)costs[i]=state.teacher_cost(rows[i].action);
        std::stable_sort(teacher.begin(),teacher.end(),[&](auto a,auto b){return costs[a]<costs[b];});
        auto add=[&](const auto& list,uint32_t count){for(auto id:list){if(std::find(selected.begin(),selected.end(),id)!=selected.end())continue;selected.push_back(id);if(!--count)break;}};
        add(teacher,8);add(shuffled,4);
        std::vector<float> logits;
        if(policy){std::vector<float> x;for(auto& row:rows)x.insert(x.end(),row.x.begin(),row.x.end());logits=policy->predict(x);auto order=teacher;std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return logits[a*3]>logits[b*3];});add(order,4);}else add(shuffled,4);
        auto first=data.labels.size();json state_trace={{"revision",step},{"triangles",state.view().triangles()},{"legal_actions",rows.size()},{"queries",json::array()}};
        std::vector<uint32_t> safe;double best=INFINITY;
        for(auto i:selected){auto candidate=state.trial(rows[i].action);auto a=evaluate_cuda(source,candidate.view(source),bounds,e),b=evaluate_cuda(source,candidate.view(source),bounds,e);++queries;
            if(!a.complete||!b.complete||a.resource_limited||b.resource_limited){write_json(output/"incomplete.json",{{"state",state_trace},{"queries",queries},{"cancelled",cancel()},{"resource_limited",a.resource_limited||b.resource_limited}});throw std::runtime_error("action audit incomplete; no labels or score assigned");}
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
    bool complete=!cancel()&&(data.states()==states||exhausted);save_actions(output/"actions.bin",data);
    auto final=evaluate(source,state.view(),bounds,e);complete&=final.complete&&final.passed;
    write_json(output/"trajectory.json",trace);write_json(output/"index.json",{{"schema",action_schema},{"complete",complete},{"asset",argv[1]},{"category",metadata.at("category")},{"contract_sha256",file_sha256(output/"contract.json")},{"path","actions.bin"},{"sha256",file_sha256(output/"actions.bin")},{"states",data.states()},{"queries",queries},{"accepted",accepted},{"source_triangles",source.triangles()},{"teacher_triangles",state.view().triangles()},{"reference_confirmed",final.complete&&final.passed},{"seconds",seconds()},{"training_started",false}});
    return complete?0:2;
}catch(const std::exception& e){std::cerr<<"action preparation: "<<e.what()<<'\n';return 1;}}
