#include "neural_placement_teacher.hpp"
#include "neural_action_gpu.hpp"
#include "neural_memory.hpp"
#include "neural_json.hpp"
#include <csignal>
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
int main(int argc,char** argv){try{
    if(argc<3)throw std::invalid_argument("blitz-neural-placement-prepare ASSET OUTPUT [--states N] [--pool N] [--pixels N] [--previous-steps N] [--source-limit N] [--adjacent-limit N] [--minutes N] [--gpu-memory-mib N] [--model MODEL] [--seed N]");
    uint32_t states=32,pool=16,previous_steps=0,seed=101;double pixels=128,minutes=5,source_limit=3,adjacent_limit=3;NeuralOptions options;fs::path model;bool sparse=false;
    auto integer=[](const char* value){std::string s=value;if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid integer option");auto x=std::stoull(s);if(x>65536)throw std::invalid_argument("integer option too large");return uint32_t(x);};
    for(int i=3;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing preparation option");std::string k=argv[i];if(k=="--states")states=integer(argv[i+1]);else if(k=="--seed")seed=integer(argv[i+1]);else if(k=="--pool")pool=integer(argv[i+1]);else if(k=="--previous-steps")previous_steps=integer(argv[i+1]);else if(k=="--pixels")pixels=std::stod(argv[i+1]);else if(k=="--source-limit")source_limit=std::stod(argv[i+1]);else if(k=="--adjacent-limit")adjacent_limit=std::stod(argv[i+1]);else if(k=="--raster-backend")options.raster_backend=raster_option(argv[i+1]);else if(k=="--vertex-storage")options.vertex_storage=storage_option(argv[i+1]);else if(k=="--audit-mode"){std::string mode=argv[i+1];if(mode!="sparse"&&mode!="exact")throw std::invalid_argument("audit mode must be sparse or exact");sparse=mode=="sparse";}else if(k=="--model")model=argv[i+1];else if(k=="--minutes")minutes=std::stod(argv[i+1]);else if(k=="--gpu-memory-mib")options.memory_mib=integer(argv[i+1]);else throw std::invalid_argument("unknown preparation option");}
    if(!states||states>4096||previous_steps>4096||!pool||pool>16||!std::isfinite(pixels)||pixels<16||pixels>512||!std::isfinite(minutes)||minutes<=0||minutes>50||!std::isfinite(source_limit)||source_limit<=0||source_limit>16||!std::isfinite(adjacent_limit)||adjacent_limit<=0||adjacent_limit>16)throw std::invalid_argument("preparation bounds");
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);fs::path output=argv[2];
    std::string policy_hash;std::unique_ptr<ActionCuda> policy;if(!model.empty()){auto w=load_weights(model);if(w.architecture!=placement_schema)throw std::invalid_argument("placement teacher requires a v3 policy");policy_hash=sha256(std::as_bytes(std::span(w.values)));policy=std::make_unique<ActionCuda>(w,options);}
    PlacementRequest request{states,pool,previous_steps,seed,pixels,minutes,source_limit,adjacent_limit,sparse,policy_hash,[]{return bool(stopped);}};
    return prepare_placements(argv[1],output,request,options,policy.get()).complete?0:2;
}catch(const std::exception& e){std::cerr<<"placement preparation: "<<e.what()<<'\n';return 1;}}
