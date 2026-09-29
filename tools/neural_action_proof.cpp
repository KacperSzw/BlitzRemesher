#include "neural_action_data.hpp"
#include <random>
#include <iostream>
#include <csignal>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
int main(int argc,char** argv){try{
    if(argc<4)throw std::invalid_argument("blitz-neural-action-proof MODEL DATASET REPORT [--minutes N]");
    double minutes=5;if(argc==6&&std::string_view(argv[4])=="--minutes")minutes=std::stod(argv[5]);else if(argc!=4)throw std::invalid_argument("invalid proof options");
    if(!std::isfinite(minutes)||minutes<=0||minutes>50)throw std::invalid_argument("invalid proof deadline");std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
    fs::path directory=argv[2],report=argv[3];auto index=read_json(directory/"index.json"),contract=read_json(directory/"contract.json");
    if(!index.at("complete").get<bool>()||index.at("sha256")!=file_sha256(directory/"actions.bin")||index.at("contract_sha256")!=file_sha256(directory/"contract.json"))throw std::invalid_argument("incomplete or changed proof dataset");
    auto data=load_actions(directory/"actions.bin");auto [mesh,metadata]=training_mesh(index.at("asset"));if(metadata!=contract.at("asset"))throw std::invalid_argument("proof source manifest differs");
    auto weights=load_weights(argv[1]);ActionCuda model(weights,{});auto output=model.predict(data.x);uint32_t correct=0,eligible=0;
    for(size_t s=0;s<data.states();++s){auto best=data.offsets[s];bool has=false;for(auto i=data.offsets[s];i<data.offsets[s+1];++i){has|=bool(data.labels[i]&Preferred);if(output[i*3]>output[best*3])best=i;}if(has){++eligible;correct+=bool(data.labels[best]&Preferred);}}
    auto start=std::chrono::steady_clock::now();auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto cancel=[&]{return bool(stopped)||seconds()>minutes*60;};
    auto e=action_eval(contract.at("pixels"));e.cancelled=cancel;auto source=mesh.view();auto bounds=blitz::bounds(source);auto c=condition(e,e.limit,.1);
    auto budget=index.at("queries").get<uint32_t>();auto target=index.at("teacher_triangles").get<size_t>();auto teacher_reduction=source.triangles()-target;
    json result={{"schema",action_schema},{"training_started",false},{"model_sha256",file_sha256(argv[1])},{"dataset_sha256",file_sha256(directory/"actions.bin")},{"contract_sha256",file_sha256(directory/"contract.json")},{"binary_sha256",file_sha256("/proc/self/exe")},{"preferred_membership",eligible?double(correct)/eligible:0},{"preferred_states",eligible},{"teacher_reduction",teacher_reduction},{"trial_budget",budget},{"rows",json::array()},{"complete",false},{"proof_passed",false}};
    // Each control uses the same legal actions, audit references and trial budget.
    // QEM is a separately named control; it never rescues neural inference.
    for(std::string method:{"neural","constant","shortest","qem","shuffled-1","shuffled-2","shuffled-3"}){
        if(cancel())break;std::mt19937 rng(method.back()>='1'&&method.back()<='3'?uint32_t(method.back()-'0'):1);auto began=seconds();
        auto rank=[&](const ActionState& state,std::span<const ActionRecord> actions){std::vector<float> scores(actions.size());
            if(method=="neural"||method.starts_with("shuffled")){std::vector<float> x;for(auto& a:actions)x.insert(x.end(),a.x.begin(),a.x.end());auto y=model.predict(x);for(size_t i=0;i<scores.size();++i)scores[i]=y[i*3];if(method.starts_with("shuffled"))std::shuffle(scores.begin(),scores.end(),rng);}
            else for(size_t i=0;i<actions.size();++i)if(method=="shortest")scores[i]=-actions[i].x[59];else if(method=="qem")scores[i]=float(-state.teacher_cost(actions[i].action));return scores;};
        auto gate=[&](MeshView candidate){auto a=evaluate_cuda(source,candidate,bounds,e);if(!a.complete||!a.passed)return false;auto b=evaluate_cuda(source,candidate,bounds,e);return b.complete&&b.passed;};
        ActionStats stats;auto lod=execute_actions(source,c,target,budget,rank,gate,&stats,cancel);auto final=evaluate(source,lod.view(source),bounds,e);bool complete=!cancel()&&final.complete;size_t reduction=source.triangles()-lod.view(source).triangles();
        result["rows"].push_back({{"method",method},{"complete",complete},{"passed",final.passed&&final.complete},{"triangles",lod.view(source).triangles()},{"reduction",reduction},{"teacher_reduction_fraction",teacher_reduction?double(reduction)/teacher_reduction:0},{"trials",stats.trials},{"ranked",stats.ranked},{"accepted",stats.accepted},{"seconds",seconds()-began},{"error",final.error},{"area",final.changed_area}});
        write_json(report,result);if(!complete)break;
    }
    result["complete"]=result["rows"].size()==7&&!cancel();if(result["complete"]==true){const auto& row=result["rows"][0];result["proof_passed"]=eligible>0&&double(correct)/eligible>=.95&&teacher_reduction>0&&row.at("passed").get<bool>()&&row.at("teacher_reduction_fraction").get<double>()>=.9;}
    result["seconds"]=seconds();write_json(report,result);std::cout<<result.dump(2)<<'\n';return result["complete"]==true?0:2;
}catch(const std::exception& e){std::cerr<<"action proof: "<<e.what()<<'\n';return 1;}}
