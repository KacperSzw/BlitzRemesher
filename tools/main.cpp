#include "blitz/io.hpp"
#include "neural_json.hpp"
#include "neural_audit_io.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <csignal>
using namespace blitz;
int benchmark_main(int,char**);
int corpus_check(const std::filesystem::path&,const std::filesystem::path&);
static volatile std::sig_atomic_t stopped=0;
static void stop(int){stopped=1;}
int main(int argc,char** argv) {
    try {
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        if(argc<2){std::cout<<"blitz defaults | info MESH | simplify MESH --config JSON --out DIR | evaluate SOURCE CANDIDATE --config JSON --pixels N --limit N | audit-replay FILE [--gpu-memory-mib N] [--device N] | corpus-check MANIFEST REPORT | bench MANIFEST CONFIG OUTPUT [--split development] [--limit N] [--minutes 50]\n";return 0;}
        std::string command=argv[1];
        if(command=="defaults"){std::cout<<settings_json(Settings{}).dump(2)<<'\n';return 0;}
        if(command=="bench")return benchmark_main(argc-2,argv+2);
        if(command=="audit-replay"){
            if(argc<3)throw std::invalid_argument("audit-replay FILE [--gpu-memory-mib N] [--device N]");NeuralOptions o;
            for(int i=3;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("option needs a value");std::string k=argv[i];if(k=="--gpu-memory-mib")o.memory_mib=neural_unsigned(argv[i+1]);else if(k=="--device")o.device=std::stoi(argv[i+1]);else throw std::invalid_argument("unknown replay option");}
            auto report=replay_audit(argv[2],o);std::cout<<report.dump(2)<<'\n';return report.at("reproduced").get<bool>()?0:2;
        }
        if(command=="corpus-check"){if(argc!=4)throw std::invalid_argument("corpus-check MANIFEST REPORT");return corpus_check(argv[2],argv[3]);}
        if(argc<3)throw std::invalid_argument("missing input");
        if(command=="info"){auto m=load_mesh(argv[2]);auto b=bounds(m.view());std::cout<<nlohmann::json({{"vertices",m.positions.size()},{"triangles",m.view().triangles()},{"diameter",b.diameter()},{"normals",!m.normals.empty()},{"uv",!m.uv.empty()},{"colors",!m.colors.empty()}}).dump(2)<<'\n';return 0;}
        Settings s;std::filesystem::path out="output";double pixels=32,limit=2;std::string neural_file;NeuralOptions neural_options;
        int start=command=="evaluate"?4:3;
        for(int i=start;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("option needs a value");std::string k=argv[i];
            if(k=="--config"||k=="--legacy-config"){nlohmann::json j;std::ifstream f(argv[i+1]);f>>j;s=settings_json(j,k=="--legacy-config");}
            else if(k=="--neural-model")neural_file=argv[i+1];else if(k=="--device")neural_options.device=std::stoi(argv[i+1]);else if(k=="--gpu-memory-mib")neural_options.memory_mib=std::stoul(argv[i+1]);
            else if(k=="--action-trials")neural_options.action_trials=neural_unsigned(argv[i+1]);else if(k=="--neural-control")neural_options.ranking=ranking_option(argv[i+1]);else if(k=="--ranking-seed")neural_options.ranking_seed=neural_unsigned(argv[i+1]);
            else if(k=="--action-batch")neural_options.action_batch=neural_batch(argv[i+1]);
            else if(k=="--neural-confirmation")neural_options.confirmation=confirmation_option(argv[i+1]);
            else if(k=="--raster-backend")neural_options.raster_backend=raster_option(argv[i+1]);
            else if(k=="--vertex-storage")neural_options.vertex_storage=storage_option(argv[i+1]);
            else if(k=="--out")out=argv[i+1];else if(k=="--pixels")pixels=std::stod(argv[i+1]);else if(k=="--limit")limit=std::stod(argv[i+1]);else throw std::invalid_argument("unknown option "+k);}
        if(!neural_file.empty())s.research.chain=ChainMode::Direct;
        s.cancelled=[]{return stopped!=0;};auto m=load_mesh(argv[2]);
        if(command=="simplify") {
            neural_options.capture_confirmation_failure=true;
            PerformanceStats work;s.performance=&work;
            auto begin=std::chrono::steady_clock::now();std::unique_ptr<NeuralModel> model;if(!neural_file.empty())model=std::make_unique<NeuralModel>(neural_file.c_str(),neural_options);NeuralStats neural_stats;Result r;
            try{r=model?generate_neural(m.view(),s,*model,&neural_stats):generate(m.view(),s);}
            catch(...){save_audit_failure(neural_stats,out/"confirmation-failure.json");throw;}
            auto generated=std::chrono::steady_clock::now();save_chain(r,out);auto exported=std::chrono::steady_clock::now();
            save_audit_failure(neural_stats,out/"confirmation-failure.json");
            auto j=result_json(r);if(model){j["method"]=neural_options.ranking==NeuralRanking::Learned?"neural":"neural-control-"+std::string(ranking_name(neural_options.ranking));j["model_sha256"]=model->sha256();j["neural_options"]=neural_json(neural_options);j["neural"]=neural_json(neural_stats);}j["seconds"]=std::chrono::duration<double>(exported-begin).count();j["generation_seconds"]=std::chrono::duration<double>(generated-begin).count();j["export_seconds"]=std::chrono::duration<double>(exported-generated).count();j["stage_seconds"]={{"reduction",work.reduction_ns*1e-9},{"raster",work.raster_ns*1e-9},{"distance",work.distance_ns*1e-9}};j["output"]=out.string();std::cout<<j.dump(2)<<'\n';return r.status==Status::Complete?0:2;
        }
        if(command=="evaluate") {
            if(argc<4)throw std::invalid_argument("evaluate requires two inputs");auto candidate=load_mesh(argv[3]);
            EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.views=s.audit_views;e.supersample=s.audit_supersample;e.max_supersample=s.max_supersample;
            e.screen_size=pixels;e.limit=limit;e.max_changed_area=s.max_changed_area;e.cancelled=s.cancelled;e.force_scalar=s.force_scalar;
            auto v=evaluate(m.view(),candidate.view(),bounds(m.view()),e);
            std::cout<<nlohmann::json({{"passed",v.passed},{"complete",v.complete},{"error_px",v.error},{"coverage_upper_px",v.coverage_upper},
                {"changed_area",v.changed_area},{"changed_area_worst_view",v.changed_area_worst_view},{"worst_view",v.worst_view},{"views",v.views_evaluated}}).dump(2)<<'\n';return v.passed?0:2;
        }
        throw std::invalid_argument("unknown command");
    }catch(const std::exception& e){std::cerr<<"blitz: "<<e.what()<<'\n';return 1;}
}
