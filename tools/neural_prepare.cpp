#include "neural_data.hpp"
#include <iostream>
#include <numeric>
#include <set>
#include <csignal>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
static volatile std::sig_atomic_t stopped=0;static void stop(int){stopped=1;}
int main(int argc,char** argv){try {
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
    if(argc<4)throw std::invalid_argument("blitz-neural-prepare CORPUS PILOT OUTPUT [--limit N] [--minutes 50]");
    fs::path corpus=argv[1],pilot=argv[2],output=argv[3],model;size_t limit=SIZE_MAX;double minutes=50;
    for(int i=4;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("missing option");std::string k=argv[i];if(k=="--limit")limit=std::stoull(argv[i+1]);else if(k=="--minutes")minutes=std::stod(argv[i+1]);else if(k=="--model")model=argv[i+1];else throw std::invalid_argument("unknown option");}
    if(!(minutes>0&&minutes<=50))throw std::invalid_argument("preparation segments must be at most 50 minutes");
    if(!neural_available())throw NeuralUnavailable("CUDA is required for teacher audits");fs::create_directories(output);
    auto start=std::chrono::steady_clock::now();auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
    auto cancelled=[&]{return stopped||elapsed()>=minutes*60;};
    auto pilot_json=read_json(pilot),corpus_json=read_json(corpus);
    std::set<std::string> excluded,groups;for(auto& a:pilot_json["assets"]){excluded.insert(a.at("id"));groups.insert(a.at("source_group"));}
    json selected=json::array();for(auto& a:corpus_json["assets"])if(a.at("split")=="development"&&!excluded.contains(a.at("id"))&&!groups.contains(a.at("source_group"))&&selected.size()<limit)selected.push_back(a);
    json contract={{"version",1},{"source_manifest_sha256",file_sha256(corpus)},{"pilot_sha256",file_sha256(pilot)},{"protocol_sha256",file_sha256("research/PROTOCOL.md")},
        {"binary_sha256",file_sha256("/proc/self/exe")},{"assets",selected},{"fractions",{.1,.25,.5,.75}},{"audited_pixels",{16,32,64}},{"views",{3,1}},{"supersample",2},{"max_supersample",4},{"profile","attributes"},{"limit",3.0},{"max_changed_area",.5}};
    WeightsData model_weights;if(!model.empty()){model_weights=load_weights(model);contract["model_sha256"]=file_sha256(model);}
    auto serialized=contract.dump();auto hash=sha256({reinterpret_cast<const std::byte*>(serialized.data()),serialized.size()});contract["hash"]=hash;
    if(fs::exists(output/"contract.json")&&read_json(output/"contract.json")!=contract)throw std::invalid_argument("preparation contract changed; choose a new output directory");write_json(output/"contract.json",contract);
    json index={{"contract_sha256",hash},{"complete",false},{"assets",json::array()}};bool failures=false;
    for(auto& a:selected) {
        std::string id=a.at("id");fs::path shard=output/(id+".bin"),row_path=output/(id+".json");
        if(fs::exists(row_path)){auto row=read_json(row_path);if(row.at("contract_sha256")==hash&&row.value("complete",false)&&file_sha256(shard)==row.at("sha256").get<std::string>()){index["assets"].push_back(row);continue;}}
        if(cancelled())break;auto asset_start=elapsed();json row={{"id",id},{"contract_sha256",hash},{"complete",false},{"source_files",a.at("files")}};
        try {
            for(auto& file:a.at("files"))if(file_sha256(file.at("path").get<std::string>())!=file.at("sha256").get<std::string>())throw std::runtime_error("source hash changed");
            auto mesh=load_mesh(a.at("path").get<std::string>());auto initial=graph(mesh.view());for(auto& i:mesh.indices)i=initial.canonical[i];compact(mesh);
            Asset data;data.graph=graph(mesh.view());auto source=mesh.view();std::vector<Lod> candidates;std::vector<Example> labels;
            auto example=[&](const Lod& l,std::vector<uint32_t> representatives,double fraction,double pixels){
                Example e;EvalSettings config;config.profile=Profile::Attributes;config.limit=3;config.screen_size=pixels;config.max_changed_area=.5;e.condition=condition(config,3,fraction);
                e.representative=std::move(representatives);e.retained.resize(data.graph.size());for(auto i:l.data.indices)e.retained[i]=1;
                for(size_t i=0;i<data.graph.size();++i)if(data.graph.flags[i]&Locked){e.retained[i]=1;e.representative[i]=uint32_t(i);}return e;
            };
            for(double fraction:{.1,.25,.5,.75}) {
                ReduceSettings rs;rs.output=OutputMode::Reuse;rs.target_triangles=std::max<size_t>(1,size_t(source.triangles()*fraction));rs.cancelled=cancelled;
                std::vector<uint32_t> representatives;auto lod=teacher(source,rs,representatives);if(cancelled())throw std::runtime_error("preparation segment interrupted");
                labels.push_back(example(lod,std::move(representatives),fraction,32));candidates.push_back(std::move(lod));
            }
            data.examples=labels;std::vector<float> embedding;if(!model.empty())embedding=encode_mesh_cuda(data.graph,model_weights,{},cancelled);auto b=bounds(source);json audits=json::array();
            for(double pixels:{16.,32.,64.}) {
                EvalSettings e;e.profile=Profile::Attributes;e.screen_size=pixels;e.limit=3;e.max_changed_area=.5;e.views={3,1,0xB1172024};e.supersample=2;e.max_supersample=4;e.cancelled=cancelled;
                size_t best=SIZE_MAX,best_triangles=source.triangles();json trials=json::array();
                std::optional<Example> neural_best;
                for(size_t i=0;i<candidates.size();++i) {auto m=evaluate_cuda(source,candidates[i].view(source),b,e);trials.push_back({{"triangles",candidates[i].view(source).triangles()},{"passed",m.passed},{"error",m.error},{"changed_area",m.changed_area},{"complete",m.complete}});
                    if(m.passed&&m.complete&&candidates[i].view(source).triangles()<best_triangles){best=i;best_triangles=candidates[i].view(source).triangles();}if(cancelled())throw std::runtime_error("preparation segment interrupted");}
                if(!model.empty())for(double fraction:{.05,.1,.25,.5,.75}) {
                    auto prediction=predict_cuda(embedding,condition(e,3,fraction),model_weights,{});std::vector<uint32_t> representatives;
                    auto candidate=decode(source,data.graph,prediction,std::max<size_t>(1,size_t(source.triangles()*fraction)),OutputMode::Reuse,nullptr,cancelled,&representatives);
                    auto measured=evaluate_cuda(source,candidate.view(source),b,e);trials.push_back({{"origin","neural"},{"triangles",candidate.view(source).triangles()},{"passed",measured.passed},{"error",measured.error},{"changed_area",measured.changed_area}});
                    if(measured.complete&&measured.passed&&candidate.view(source).triangles()<best_triangles){best_triangles=candidate.view(source).triangles();neural_best=example(candidate,std::move(representatives),fraction,pixels);}
                    if(cancelled())throw std::runtime_error("refinement segment interrupted");
                }
                Example accepted;if(neural_best)accepted=std::move(*neural_best);else if(best!=SIZE_MAX)accepted=labels[best];else {std::vector<uint32_t> identity(data.graph.size());std::iota(identity.begin(),identity.end(),0);Lod same;same.data.indices=mesh.indices;accepted=example(same,std::move(identity),1,pixels);}
                accepted.condition=condition(e,3,double(best_triangles)/source.triangles());data.examples.push_back(std::move(accepted));audits.push_back({{"pixels",pixels},{"selected_triangles",best_triangles},{"fallback",best==SIZE_MAX&&!neural_best},{"neural_improvement",bool(neural_best)},{"trials",trials}});
            }
            save_asset(shard,data);row["complete"]=true;row["path"]=shard.filename().string();row["sha256"]=file_sha256(shard);row["vertices"]=data.graph.size();row["triangles"]=source.triangles();row["examples"]=data.examples.size();row["audits"]=audits;row["seconds"]=elapsed()-asset_start;
            index["assets"].push_back(row);std::cout<<json({{"asset",id},{"seconds",row["seconds"]},{"vertices",row["vertices"]}}).dump()<<std::endl;
        }catch(const std::exception& e){row["failure"]=e.what();failures=true;std::cerr<<id<<": "<<e.what()<<'\n';}
        write_json(row_path,row);write_json(output/"index.json",index);if(cancelled())break;
    }
    index["complete"]=index["assets"].size()==selected.size();index["expected"]=selected.size();index["seconds_this_segment"]=elapsed();write_json(output/"index.json",index);
    std::cout<<json({{"complete",index["complete"]},{"assets",index["assets"].size()},{"expected",selected.size()},{"seconds",elapsed()}}).dump()<<'\n';return index["complete"]==true?0:cancelled()?2:failures?1:2;
}catch(const std::exception& e){std::cerr<<"neural preparation: "<<e.what()<<'\n';return 1;}}
