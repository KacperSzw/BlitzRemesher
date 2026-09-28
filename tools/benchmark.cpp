#include "blitz/io.hpp"
#include <openssl/evp.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <regex>
#include <sstream>
#include <thread>
#if defined(__unix__)
#include <sys/resource.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <spawn.h>
#include <signal.h>
#include <unistd.h>
extern char** environ;
#endif
using namespace blitz;
using json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
std::string digest(std::span<const std::byte> bytes) {
    unsigned char out[EVP_MAX_MD_SIZE];unsigned n{};
    EVP_Digest(bytes.data(),bytes.size(),out,&n,EVP_sha256(),nullptr);
    std::ostringstream s;for(unsigned i=0;i<n;++i)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(out[i]);return s.str();
}
std::string digest(const std::string& text){return digest({reinterpret_cast<const std::byte*>(text.data()),text.size()});}
std::string file_hash(const fs::path& p) {
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot open "+p.string());
    auto ctx=EVP_MD_CTX_new();EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr);
    char buf[65536];while(f){f.read(buf,sizeof(buf));EVP_DigestUpdate(ctx,buf,size_t(f.gcount()));}
    unsigned char out[EVP_MAX_MD_SIZE];unsigned n{};EVP_DigestFinal_ex(ctx,out,&n);EVP_MD_CTX_free(ctx);
    std::ostringstream s;for(unsigned i=0;i<n;++i)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(out[i]);return s.str();
}
json read(const fs::path& p){std::ifstream f(p);json j;f>>j;return j;}
void write(const fs::path& p,const json& j){if(!p.parent_path().empty())fs::create_directories(p.parent_path());auto part=p;part+=".part";{std::ofstream f(part);f<<j.dump(2)<<'\n';if(!f)throw std::runtime_error("cannot checkpoint");}fs::rename(part,p);}
long rss() {
#if defined(__unix__)
    rusage r{};getrusage(RUSAGE_SELF,&r);
#if defined(__APPLE__)
    return r.ru_maxrss/1024;
#else
    return r.ru_maxrss;
#endif
#else
    return 0;
#endif
}
std::string attribute_hash(MeshView m) {
    auto stream=[](auto s) {
        using T=decltype(s[0]);
        if(!s.count)return digest(std::string{});
        if(s.stride==sizeof(T))return digest({s.data,s.count*sizeof(T)});
        std::string hashes;for(size_t i=0;i<s.count;++i){auto value=s[i];hashes+=digest({reinterpret_cast<const std::byte*>(&value),sizeof(T)});}return digest(hashes);
    };
    return digest(stream(m.positions)+stream(m.normals)+stream(m.uv)+stream(m.colors)+stream(m.tangents)
        +digest(std::as_bytes(m.materials))+digest(std::as_bytes(m.double_sided)));
}
std::set<std::string> subset(const std::vector<std::pair<std::string,size_t>>& groups,size_t n,const std::set<std::string>& exclude) {
    std::vector<std::optional<std::vector<std::string>>> dp(n+1);dp[0]=std::vector<std::string>{};
    for(auto& [id,count]:groups)if(!exclude.contains(id)&&count<=n)
        for(size_t i=n+1;i-->count;)if(!dp[i]&&dp[i-count]){auto v=*dp[i-count];v.push_back(id);dp[i]=std::move(v);}
    if(!dp[n])throw std::runtime_error("cannot meet grouped split quota");return {dp[n]->begin(),dp[n]->end()};
}
void child(std::vector<std::string> args,std::chrono::steady_clock::time_point deadline) {
#if defined(__unix__)
    std::vector<char*> pointers;for(auto& a:args)pointers.push_back(a.data());pointers.push_back(nullptr);
    pid_t pid;if(posix_spawn(&pid,pointers[0],nullptr,nullptr,pointers.data(),environ))throw std::runtime_error("cannot start baseline");
    int status=0;
    for(;;) {
        auto rc=waitpid(pid,&status,WNOHANG);
        if(rc==pid)break;
        if(rc<0)throw std::runtime_error("cannot wait for baseline");
        if(std::chrono::steady_clock::now()>=deadline){kill(pid,SIGKILL);waitpid(pid,&status,0);throw std::runtime_error("baseline exceeded batch deadline");}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if(!WIFEXITED(status)||WEXITSTATUS(status)!=0)throw std::runtime_error("baseline rejected mesh or failed (exit "+std::to_string(status)+")");
#else
    (void)args;(void)deadline;throw std::runtime_error("external benchmark launch currently requires POSIX");
#endif
}
}
int corpus_check(const fs::path& manifest,const fs::path& output) {
    auto input=read(manifest);json report={{"version",1},{"acquisition_sha256",file_hash(manifest)},{"assets",json::array()}};
    auto rights=read("research/scan-rights.json");
    size_t failed=0,woody=0;uint64_t bytes=0;std::map<std::string,std::string> geometry_hashes;
    for(auto asset:input["assets"]) {
        try {
            if(asset.value("provider","")=="smithsonian") {
                auto identity=asset.at("source_identity").get<std::string>();
                if(!rights["items"].contains(identity))throw std::runtime_error("scan needs item-specific media rights verification");
                asset["license_evidence"]=rights["items"][identity];asset["rights_verified_on"]=rights["verified_on"];
            }
            for(auto file:asset["files"]){fs::path p=file.at("path");bytes+=fs::file_size(p);if(file_hash(p)!=file.at("sha256").get<std::string>())throw std::runtime_error("source checksum mismatch");}
            auto m=load_mesh(asset.at("path").get<std::string>());
            std::string h=digest({reinterpret_cast<const std::byte*>(m.positions.data()),m.positions.size()*sizeof(Vec3)});
            h=digest(h+digest({reinterpret_cast<const std::byte*>(m.indices.data()),m.indices.size()*4}));
            if(geometry_hashes.contains(h))throw std::runtime_error("duplicate geometry of "+geometry_hashes[h]);
            geometry_hashes[h]=asset["id"];
            asset["geometry_sha256"]=h;asset["vertices"]=m.positions.size();asset["triangles"]=m.view().triangles();asset["diameter"]=bounds(m.view()).diameter();
            asset["normals"]=!m.normals.empty();asset["uv"]=!m.uv.empty();asset["colors"]=!m.colors.empty();asset["import_status"]="ok";
            std::string family=asset.value("provider","")=="smithsonian"?asset.value("title",std::string{}):asset.at("source_identity").get<std::string>();
            family=std::regex_replace(family,std::regex("[0-9]+"),"");family=std::regex_replace(family,std::regex("_+$| +$"),"");
            asset["source_group"]=asset.at("provider").get<std::string>()+":"+family;
            if(asset.value("woody",false))++woody;
        }catch(const std::exception& e){asset["import_status"]="failed";asset["failure"]=e.what();++failed;}
        asset.erase("source_metadata");report["assets"].push_back(asset);
        std::cerr<<"QC "<<report["assets"].size()<<"/"<<input["assets"].size()<<" "<<asset["id"]<<" "<<asset["import_status"]<<'\n';
    }
    std::map<std::string,std::map<std::string,size_t>> grouped;
    for(auto& a:report["assets"])++grouped[a.at("category")][a.value("source_group",a.at("id").get<std::string>())];
    for(auto& [category,counts]:grouped) {
        size_t total=0;std::vector<std::pair<std::string,size_t>> groups(counts.begin(),counts.end());for(auto& g:groups)total+=g.second;
        std::sort(groups.begin(),groups.end(),[](auto& a,auto& b){return digest("B1172026"+a.first)<digest("B1172026"+b.first);});
        size_t n=category=="manufactured"?7:category=="stress"?3:5;
        if(total<2*n)throw std::runtime_error("incomplete category quota");
        auto test=subset(groups,n,{}),validation=subset(groups,n,test);
        for(auto& a:report["assets"])if(a["category"]==category){auto id=a.at("source_group").get<std::string>();a["split"]=test.contains(id)?"held_out":validation.contains(id)?"validation":"development";}
    }
    report["valid"]=failed==0&&report["assets"].size()==120&&woody>=10&&bytes<=50ull*1024*1024*1024;
    report["failed"]=failed;report["woody"]=woody;report["bytes"]=bytes;report["grouping"]="Identity/name families with numeric variants grouped; exact geometry duplicates rejected. Near-duplicate semantic review remains a human research task.";
    write(output,report);std::cout<<json({{"valid",report["valid"]},{"assets",report["assets"].size()},{"failed",failed},{"bytes",bytes},{"woody",woody}}).dump(2)<<'\n';return report["valid"]==true?0:2;
}
int benchmark_main(int argc,char** argv) {
    if(argc<3)throw std::invalid_argument("bench MANIFEST CONFIG OUTPUT");
    fs::path manifest=argv[0],config=argv[1],output=argv[2],baseline_dir="build/research",build_stamp="research/build.json";std::string split="development",method="native";size_t limit=SIZE_MAX;double minutes=50;
    for(int i=3;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("missing benchmark option value");std::string k=argv[i];
        if(k=="--split")split=argv[i+1];else if(k=="--limit")limit=std::stoull(argv[i+1]);else if(k=="--minutes")minutes=std::stod(argv[i+1]);
        else if(k=="--baseline")method=argv[i+1];else if(k=="--baseline-dir")baseline_dir=argv[i+1];else if(k=="--build-stamp")build_stamp=argv[i+1];else throw std::invalid_argument("unknown benchmark option");}
    if(!(minutes>0&&minutes<=50))throw std::invalid_argument("batch time must be <=50 minutes");
    auto corpus=read(manifest);
    if(!corpus.value("benchmark_eligible",true))throw std::invalid_argument("collection-only assets: opacity-aware benchmarking is deferred");
    auto settings=settings_json(read(config));auto normalized=settings_json(settings);
    auto storage=reduction_storage();
    json metadata={{"version",2},{"input_format","linear-rgba8-v2"},{"quadric_bytes",storage.quadric_bytes},{"candidate_bytes",storage.candidate_bytes},
      {"packed_coverage",packed_coverage_enabled()},{"stage_timing",true},{"manifest_sha256",file_hash(manifest)},{"config",normalized},{"config_sha256",digest(normalized.dump())},
      {"protocol_sha256",file_hash("research/PROTOCOL.md")},{"compiler",__VERSION__},{"split",split},{"limit",limit},{"threads",1},
      {"backend",evaluator_backend(settings.force_scalar)},{"peak_rss_scope","process high-water; KiB"}};
    metadata["camera_sha256"]=digest(normalized["search_views"].dump()+normalized["audit_views"].dump());
    fs::path baseline;
    metadata["method"]=method;
    metadata["output_hash_scope"]="output_sha256: owned positions and indices; attributes_sha256: all output streams, including shared source data";
    if(method!="native") {
        if(method!="meshopt"&&method!="fastquadric"&&method!="cgal-lt"&&method!="cgal-qem"&&method!="cgal-probabilistic")throw std::invalid_argument("unknown baseline");
        baseline=fs::absolute(baseline_dir/("blitz-baseline-"+(method.starts_with("cgal-")?std::string("cgal"):method)));
        metadata["baseline_sha256"]=file_hash(baseline);
        metadata["baseline_reference"]=method=="meshopt"?"9e1f07b159d3cb777f1c67ed31fc11fd117986f4":method=="fastquadric"?"65df07dc54766e3ee480482f1c881a62767831cc":"CGAL 6.1.1 in flake.lock; exact executable identified by SHA-256";
    }
#if defined(__linux__)
    metadata["binary_sha256"]=file_hash("/proc/self/exe");
    std::ifstream cpu("/proc/cpuinfo");std::string line;while(std::getline(cpu,line))if(line.starts_with("model name")){metadata["cpu"]=line;break;}
#endif
    if(fs::exists(build_stamp))metadata["build"]=read(build_stamp);
    else if(build_stamp!="research/build.json")throw std::invalid_argument("explicit build stamp is missing");
    if(metadata.contains("binary_sha256")&&metadata.contains("build")&&metadata["build"].contains("binary_sha256")
       &&metadata["build"]["binary_sha256"]!=metadata["binary_sha256"])throw std::invalid_argument("build stamp belongs to a different executable");
    auto runhash=digest(metadata.dump());metadata["run_sha256"]=runhash;
    fs::create_directories(output/"rows");
    if(fs::exists(output/"metadata.json")&&read(output/"metadata.json")!=metadata)throw std::runtime_error("resume refused: input, settings, protocol or binary changed");
    write(output/"metadata.json",metadata);
    auto start=std::chrono::steady_clock::now(),deadline=start+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(minutes*60));
    std::vector<json> assets;
    for(auto a:corpus["assets"])if(split=="all"||a.at("split")==split){if(assets.size()<limit)assets.push_back(a);}
    if(assets.empty())throw std::invalid_argument("empty benchmark selection");
    size_t done=0;std::map<std::string,std::vector<double>> categories;double seconds=0;size_t fallbacks=0;
    for(auto asset:assets) {
        std::string id=asset.at("id");auto path=output/"rows"/(id+".json");json row;
        // Completed rows are reusable only while the actual bytes still match the frozen inputs.
        for(auto file:asset["files"])if(file_hash(file.at("path").get<std::string>())!=file.at("sha256").get<std::string>())throw std::runtime_error("source changed: "+id);
        if(fs::exists(path)){row=read(path);if(row.value("complete",false)==false)row=nullptr;}
        if(row.is_null()) {
            if(std::chrono::steady_clock::now()>=deadline)break;
            auto begin=std::chrono::steady_clock::now();
            row={{"id",id},{"category",asset["category"]},{"run_sha256",runhash},{"complete",false}};
            row["source_files"]=asset["files"];
            try {
                auto mesh=load_mesh(asset.at("path").get<std::string>());settings.cancelled=[&]{return std::chrono::steady_clock::now()>=deadline;};
                row["load_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                row["canonical_attributes_sha256"]=attribute_hash(mesh.view());
                Proposer proposer;
                if(method!="native") {
                    if(settings.profile!=Profile::Coverage)throw std::runtime_error("external adapters expose geometry-only coverage capabilities");
                    if(settings.research.output==OutputMode::Reuse&&method!="meshopt")throw std::runtime_error("baseline cannot preserve source vertices");
                    fs::create_directories(output/".scratch");auto scratch=output/".scratch";
                    proposer=[&,scratch](MeshView input,const ReduceSettings& rs) {
                        uint16_t material=input.material(0);
                        for(size_t f=0;f<input.triangles();++f)if(input.material(f)!=material)throw std::runtime_error("external adapter requires one material (no silent merging)");
                        save_ply(input,scratch/"input.ply");
                        std::vector<std::string> args{baseline.string(),(scratch/"input.ply").string(),(scratch/"candidate.ply").string(),std::to_string(rs.target_triangles),rs.output==OutputMode::Reuse?"reuse":"rebuild"};
                        if(method.starts_with("cgal-"))args.push_back(method.substr(5));child(std::move(args),deadline);
                        Lod l;l.data=load_mesh(scratch/"candidate.ply");l.shared_vertices=rs.output==OutputMode::Reuse;
                        l.data.materials.assign(l.data.indices.size()/3,material);l.data.double_sided.assign(input.double_sided.begin(),input.double_sided.end());
                        if(l.shared_vertices) {
                            if(l.data.positions.size()!=input.positions.count)throw std::runtime_error("baseline violated vertex reuse");
                            for(size_t i=0;i<input.positions.count;++i){auto p=input.positions[i];if(std::memcmp(&p,&l.data.positions[i],sizeof(p)))throw std::runtime_error("baseline changed source position");}
                            l.data.positions.clear();l.data.normals.clear();l.data.uv.clear();l.data.colors.clear();l.data.tangents.clear();
                        }
                        return l;
                    };
                }
                PerformanceStats work;settings.performance=&work;
                auto generation_begin=std::chrono::steady_clock::now();
                auto result=generate(mesh.view(),settings,proposer);
                row["generation_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-generation_begin).count();
                row["stage_seconds"]={{"reduction",work.reduction_ns*1e-9},{"raster",work.raster_ns*1e-9},{"distance",work.distance_ns*1e-9}};
                row["numerics"]={{"solve_attempts",work.solve_attempts},{"singular_solves",work.singular_solves},{"nonfinite_solves",work.nonfinite_solves},
                  {"position_fallbacks",work.position_fallbacks},{"nonfinite_costs",work.nonfinite_costs}};
                row["coverage_cache"]={{"rasters",work.coverage_rasters},{"fields",work.coverage_fields},{"mask_hits",work.coverage_mask_hits},
                  {"field_hits",work.coverage_field_hits},{"bypasses",work.coverage_cache_bypasses},{"peak_bytes",work.coverage_cache_peak_bytes}};
                row["result"]=result_json(result);
                row["complete"]=result.status==Status::Complete;
                double triangles=0;for(size_t i=1;i<result.lods.size();++i)triangles+=result.lods[i].data.indices.size()/3;
                row["ratio"]=triangles/((result.lods.size()-1)*mesh.view().triangles());row["fallback"]=row["ratio"]==1;
                row["final_ratio"]=double(result.lods.back().data.indices.size()/3)/mesh.view().triangles();
                size_t tail_count=std::min<size_t>(3,result.lods.size()-1);double tail=0;
                for(size_t i=result.lods.size()-tail_count;i<result.lods.size();++i)tail+=result.lods[i].data.indices.size()/3;
                row["last_three_ratio"]=tail/(tail_count*mesh.view().triangles());
                std::string output_hash;for(auto& l:result.lods){output_hash+=digest({reinterpret_cast<const std::byte*>(l.data.indices.data()),l.data.indices.size()*4});output_hash+=digest({reinterpret_cast<const std::byte*>(l.data.positions.data()),l.data.positions.size()*sizeof(Vec3)});}
                row["output_sha256"]=digest(output_hash);
                std::string attributes;for(auto& l:result.lods)attributes+=attribute_hash(l.view(result.source));row["attributes_sha256"]=digest(attributes);
                auto export_begin=std::chrono::steady_clock::now();
                if(result.status==Status::Complete)save_chain(result,output/"meshes"/id);
                row["export_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-export_begin).count();
            }catch(const std::exception& e){row["failure"]=e.what();row["ratio"]=1;row["fallback"]=true;row["complete"]=std::chrono::steady_clock::now()<deadline;row["failed"]=true;}
            row["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();row["peak_rss_kib"]=rss();write(path,row);
        }
        if(!row.value("complete",false))break;
        if(row.at("run_sha256")!=runhash)throw std::runtime_error("row hash mismatch");
        categories[row.at("category")].push_back(row.at("ratio"));seconds+=row.at("seconds").get<double>();fallbacks+=row.value("fallback",false);++done;
        std::cerr<<done<<"/"<<assets.size()<<" "<<id<<" ratio="<<row.at("ratio")<<" seconds="<<row.at("seconds")<<'\n';
    }
    json summary={{"run_sha256",runhash},{"complete",done==assets.size()},{"expected",assets.size()},{"completed",done},{"seconds",seconds},{"fallbacks",fallbacks},{"categories",json::object()}};
    double mean=0;for(auto& [name,values]:categories){double sum=0;for(auto v:values)sum+=v;double avg=sum/values.size();mean+=avg;summary["categories"][name]={{"count",values.size()},{"mean_ratio",avg}};}
    summary["score"]=done==assets.size()?json(100*(1-mean/categories.size())):json(nullptr);
    write(output/"summary.json",summary);std::cout<<summary.dump(2)<<'\n';return done==assets.size()?0:2;
}
