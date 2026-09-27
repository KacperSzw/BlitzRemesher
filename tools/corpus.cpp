#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <openssl/evp.h>
#include <archive.h>
#include <archive_entry.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <map>
#include <set>
#include <regex>
using json=nlohmann::json;
namespace fs=std::filesystem;
static constexpr uint64_t max_file=1024ull*1024*1024;
static size_t append(void* p,size_t a,size_t b,void* u) {
    auto& s=*static_cast<std::string*>(u);
    if(s.size()+a*b>max_file) return 0;
    s.append(static_cast<char*>(p),a*b); return a*b;
}
static std::string request(const std::string& url) {
    for(int attempt=0;attempt<5;++attempt) {
        CURL* c=curl_easy_init(); if(!c) throw std::runtime_error("curl init");
        std::string out;
        curl_easy_setopt(c,CURLOPT_URL,url.c_str());
        curl_easy_setopt(c,CURLOPT_USERAGENT,"BlitzRemesher/0.1 research-corpus");
        curl_easy_setopt(c,CURLOPT_FOLLOWLOCATION,1L);
        curl_easy_setopt(c,CURLOPT_PROTOCOLS_STR,"https,http");
        curl_easy_setopt(c,CURLOPT_REDIR_PROTOCOLS_STR,"https,http");
        curl_easy_setopt(c,CURLOPT_CONNECTTIMEOUT,30L);
        curl_easy_setopt(c,CURLOPT_TIMEOUT,240L);
        curl_easy_setopt(c,CURLOPT_MAXFILESIZE_LARGE,curl_off_t(max_file));
        curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,append);
        curl_easy_setopt(c,CURLOPT_WRITEDATA,&out);
        auto rc=curl_easy_perform(c); long code=0; curl_easy_getinfo(c,CURLINFO_RESPONSE_CODE,&code);
        curl_off_t retry=0; curl_easy_getinfo(c,CURLINFO_RETRY_AFTER,&retry);
        curl_easy_cleanup(c);
        if(rc==CURLE_OK && code==200) return out;
        if(code==404 || code==403) throw std::runtime_error("HTTP "+std::to_string(code)+" "+url);
        std::this_thread::sleep_for(std::chrono::seconds(std::max<curl_off_t>(retry,1<<attempt)));
    }
    throw std::runtime_error("download failed: "+url);
}
static void write(const fs::path& p,const std::string& s) {
    fs::create_directories(p.parent_path());
    auto tmp=p; tmp+=".part";
    { std::ofstream f(tmp,std::ios::binary); f.write(s.data(),s.size()); if(!f) throw std::runtime_error("write failed"); }
    fs::rename(tmp,p);
}
static std::string read(const fs::path& p) { std::ifstream f(p,std::ios::binary); return {std::istreambuf_iterator<char>(f),{}}; }
static std::string hash(const std::string& s) {
    unsigned char out[EVP_MAX_MD_SIZE]; unsigned n{};
    EVP_Digest(s.data(),s.size(),out,&n,EVP_sha256(),nullptr);
    std::ostringstream r; for(unsigned i=0;i<n;++i) r<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(out[i]); return r.str();
}
static json cached(const std::string& url,const fs::path& path) {
    if(!fs::exists(path)) write(path,request(url));
    return json::parse(read(path));
}
static bool contains(const json& a,const std::string& s) {return std::find(a.begin(),a.end(),s)!=a.end();}
static bool woody(const std::string& s) {return std::regex_search(s,std::regex("trunk|stump|root|branch|(^|_)log(_|$)|driftwood"));}
static std::string safe(std::string s) { for(char& c:s) if(!std::isalnum(static_cast<unsigned char>(c))&&c!='-'&&c!='_') c='_'; return s; }
static json poly(const std::string& id,const std::string& category,const fs::path& root) {
    auto dir=root/"polyhaven"/id;
    auto files=cached("https://api.polyhaven.com/files/"+id,dir/"files.json");
    auto info=cached("https://api.polyhaven.com/info/"+id,dir/"info.json");
    if(!files.contains("gltf")) throw std::runtime_error("no glTF");
    auto variants=files["gltf"];
    auto desc=variants.contains("1k")?variants["1k"]["gltf"]:variants.begin().value()["gltf"];
    auto path=dir/"source.gltf";
    auto text=fs::exists(path)?read(path):request(desc["url"]);
    auto doc=json::parse(text);
    uint64_t triangles=0;
    for(auto& mesh:doc.value("meshes",json::array()))for(auto& p:mesh["primitives"])
        if(p.contains("indices"))triangles+=doc["accessors"][p["indices"].get<size_t>()]["count"].get<uint64_t>()/3;
    if(triangles>2000000)throw std::runtime_error("pre-freeze corpus cap: 2 million triangles per asset");
    if(doc.contains("skins") || doc.contains("animations")) throw std::runtime_error("animated asset");
    for(auto& m:doc.value("materials",json::array()))
        if(m.value("alphaMode","OPAQUE")!="OPAQUE") throw std::runtime_error("alpha-dependent geometry");
    for(auto& m:doc.value("meshes",json::array())) for(auto& p:m["primitives"])
        if(p.contains("targets") || p.contains("extensions")) throw std::runtime_error("unsupported mesh extensions");
    json downloads=json::array();
    uint64_t total=text.size();
    for(auto& b:doc["buffers"]) {
        std::string uri=b["uri"];
        fs::path rel(uri);
        if(rel.is_absolute() || uri.find("..")!=std::string::npos) throw std::runtime_error("unsafe buffer path");
        if(uri.starts_with("data:")) throw std::runtime_error("embedded buffer unsupported by acquisition");
        if(!desc["include"].contains(uri)) throw std::runtime_error("missing buffer dependency");
        auto meta=desc["include"][uri]; uint64_t sz=meta.value("size",uint64_t(0));
        if(total+sz>max_file) throw std::runtime_error("asset exceeds size cap");
        auto bp=dir/rel; auto bytes=fs::exists(bp)?read(bp):request(meta["url"]);
        if(sz && bytes.size()!=sz) throw std::runtime_error("buffer size mismatch");
        write(bp,bytes); total+=bytes.size();
        downloads.push_back({{"path",bp.string()},{"url",meta["url"]},{"sha256",hash(bytes)},{"bytes",bytes.size()}});
    }
    write(path,text);
    downloads.push_back({{"path",path.string()},{"url",desc["url"]},{"sha256",hash(text)},{"bytes",text.size()}});
    return {{"id","ph_"+id},{"provider","polyhaven"},{"source_identity",id},{"source_group","ph_"+id},
      {"category",category},{"woody",woody(id)},{"license","CC0-1.0"},{"license_url","https://polyhaven.com/license"},
      {"source_url","https://polyhaven.com/a/"+id},{"path",path.string()},{"files",downloads},{"bytes",total},
      {"source_metadata",info},{"opaque",true}};
}
static std::string extract_obj(const std::string& bytes) {
    archive* a=archive_read_new(); archive_read_support_format_zip(a); archive_read_support_filter_all(a);
    if(archive_read_open_memory(a,bytes.data(),bytes.size())!=ARCHIVE_OK) {archive_read_free(a);throw std::runtime_error("bad ZIP");}
    archive_entry* e{}; std::string best;
    while(archive_read_next_header(a,&e)==ARCHIVE_OK) {
        fs::path name(archive_entry_pathname(e));
        auto n=archive_entry_size(e);
        if(name.extension()!=".obj" || n<=0 || uint64_t(n)>max_file || uint64_t(n)<=best.size()) {archive_read_data_skip(a);continue;}
        std::string obj(size_t(n),'\0'); size_t pos=0;
        while(pos<obj.size()) {auto got=archive_read_data(a,obj.data()+pos,obj.size()-pos);if(got<=0)break;pos+=size_t(got);}
        if(pos==obj.size()) best=std::move(obj);
    }
    archive_read_free(a); if(best.empty()) throw std::runtime_error("ZIP contains no OBJ"); return best;
}
static json smithsonian(const json& row,const fs::path& root) {
    auto c=row["content"]; std::string identity=c["model_url"],id=safe(identity);
    auto path=root/"smithsonian"/id/"source.obj";
    std::string ziphash,bytes;
    if(fs::exists(path)) bytes=read(path);
    else {auto zip=request(c["uri"]);ziphash=hash(zip);bytes=extract_obj(zip);write(path,bytes);}
    return {{"id","si_"+id},{"provider","smithsonian"},{"source_identity",identity},{"source_group","si_"+id},
      {"title",row["title"]},{"category","stress"},{"woody",false},{"license","CC0-1.0"},
      {"license_url","https://www.si.edu/openaccess"},{"source_url",c["uri"]},{"path",path.string()},
      {"files",json::array({{{"path",path.string()},{"sha256",hash(bytes)},{"bytes",bytes.size()}}})},
      {"archive_sha256",ziphash},{"bytes",bytes.size()},{"variant",c.value("quality","unknown")},{"opaque",true}};
}
int main(int argc,char** argv) try {
    fs::path root=argc>1?argv[1]:"data";
    const auto start=std::chrono::steady_clock::now();
    curl_global_init(CURL_GLOBAL_DEFAULT);
    fs::create_directories(root); auto mp=root/"manifest.json";
    if(argc>2) {
        auto frozen=json::parse(read(argv[2]));size_t complete=0;uint64_t total=0;
        for(auto& asset:frozen["assets"]) {
            for(auto& file:asset["files"]) {
                total+=file.at("bytes").get<uint64_t>();if(total>50ull*1024*1024*1024)throw std::runtime_error("frozen corpus exceeds size budget");
                fs::path original=file.at("path").get<std::string>();
                if(original.is_absolute())throw std::runtime_error("absolute corpus path");
                fs::path relative;bool first=true;for(auto part:original){if(part=="..")throw std::runtime_error("unsafe corpus path");if(first){first=false;continue;}relative/=part;}
                if(relative.empty())throw std::runtime_error("empty corpus path");auto destination=root/relative;
                auto expected=file.at("sha256").get<std::string>();
                if(fs::exists(destination)&&hash(read(destination))==expected)continue;
                if(std::chrono::steady_clock::now()-start>std::chrono::minutes(50)){std::cout<<"Replay checkpoint; rerun the same command\n";return 2;}
                std::string bytes;
                if(file.contains("url"))bytes=request(file.at("url"));
                else if(asset.at("provider")=="smithsonian")bytes=extract_obj(request(asset.at("source_url")));
                else throw std::runtime_error("missing frozen file URL");
                if(hash(bytes)!=expected)throw std::runtime_error("upstream bytes changed; refusing nonreproducible corpus");
                write(destination,bytes);
            }
            ++complete;std::cout<<"Verified "<<complete<<"/"<<frozen["assets"].size()<<std::endl;
        }
        std::cout<<"Frozen corpus replay complete\n";curl_global_cleanup();return 0;
    }
    json manifest=fs::exists(mp)?json::parse(read(mp)):json{{"version",1},{"seed",0xB1172026u},{"assets",json::array()},{"rejected",json::array()}};
    std::set<std::string> seen; std::map<std::string,size_t> counts;
    uint64_t acquired_bytes=0;
    for(auto& a:manifest["assets"]) {seen.insert(a["id"]);counts[a["category"].get<std::string>()]++;}
    for(auto& a:manifest["assets"])for(auto& f:a["files"])acquired_bytes+=f.value("bytes",uint64_t(0));
    auto checkpoint=[&]{write(mp,manifest.dump(2));};
    auto add=[&](json a) {for(auto& f:a["files"])acquired_bytes+=f.value("bytes",uint64_t(0));
      std::string cat=a["category"];counts[cat]++;seen.insert(a["id"]);manifest["assets"].push_back(std::move(a));checkpoint();
      std::cout<<manifest["assets"].size()<<" downloaded, "<<cat<<" "<<counts[cat]<<std::endl;};
    auto expired=[&]{return std::chrono::steady_clock::now()-start>std::chrono::minutes(50)||acquired_bytes+max_file>50ull*1024*1024*1024;};
    auto assets=cached("https://api.polyhaven.com/assets?t=models",root/"catalog-polyhaven.json");
    std::map<std::string,std::vector<std::string>> candidates;
    for(auto it=assets.begin();it!=assets.end();++it) {
        auto cats=it.value()["categories"]; auto id=it.key();
        if(id=="long_life_food"||id=="russian_food_cans_01"||id.starts_with("coastal_cliff"))continue; // QC: packaging and cliffs are not organic exemplars.
        if(contains(cats,"rocks")) candidates["rocks"].push_back(id);
        else if(contains(cats,"nature")||contains(cats,"food")) candidates["organic"].push_back(id);
        else candidates["manufactured"].push_back(id);
    }
    for(auto& [cat,ids]:candidates) std::stable_sort(ids.begin(),ids.end(),[&](auto& a,auto& b){
        if(cat=="organic" && woody(a)!=woody(b)) return woody(a);
        return hash("B1172026"+a)<hash("B1172026"+b);
    });
    for(auto [cat,quota]:std::map<std::string,size_t>{{"rocks",30},{"organic",30},{"manufactured",40}})
        for(auto& id:candidates[cat]) {
            if(counts[cat]>=quota||expired()) break;
            if(seen.contains("ph_"+id)) continue;
            try {add(poly(id,cat,root));}
            catch(const std::exception& e) {manifest["rejected"].push_back({{"id","ph_"+id},{"reason",e.what()}});checkpoint();std::cerr<<id<<": "<<e.what()<<'\n';}
        }
    for(unsigned page=0;counts["stress"]<20 && page<100 && !expired();++page) {
        auto url="https://3d-api.si.edu/api/v1.0/content/file/search?model_type=obj&rows=100&start="+std::to_string(page*100);
        auto list=cached(url,root/("catalog-si-"+std::to_string(page)+".json"));
        auto rows=list.value("rows",json::array()); if(rows.empty()) break;
        for(auto& row:rows) {
            auto& c=row["content"]; auto id="si_"+safe(c["model_url"]);
            if(seen.contains(id)||counts["stress"]>=20||expired()) continue;
            if(c.value("quality","")!="Low_resolution") continue;
            try {add(smithsonian(row,root));}
            catch(const std::exception& e) {manifest["rejected"].push_back({{"id",id},{"reason",e.what()}});checkpoint();std::cerr<<id<<": "<<e.what()<<'\n';}
        }
    }
    // Deterministic category-stratified assignment; frozen only once all quotas exist.
    bool complete=counts["rocks"]==30&&counts["organic"]==30&&counts["manufactured"]==40&&counts["stress"]==20;
    size_t woody_count=0;for(auto& a:manifest["assets"])woody_count+=a.value("woody",false);complete=complete&&woody_count>=10;
    if(complete) {
        for(auto cat:{"rocks","organic","manufactured","stress"}) {
            std::vector<size_t> order;
            for(size_t i=0;i<manifest["assets"].size();++i) if(manifest["assets"][i]["category"]==cat) order.push_back(i);
            std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return hash("split"+manifest["assets"][a]["source_group"].get<std::string>())<hash("split"+manifest["assets"][b]["source_group"].get<std::string>());});
            size_t train=std::string(cat)=="manufactured"?26:std::string(cat)=="stress"?14:20;
            size_t val=std::string(cat)=="manufactured"?7:std::string(cat)=="stress"?3:5;
            for(size_t j=0;j<order.size();++j) manifest["assets"][order[j]]["split"]=j<train?"development":j<train+val?"validation":"held-out";
        }
    }
    manifest["acquisition_complete"]=complete; checkpoint(); curl_global_cleanup();
    std::cout<<"Acquisition "<<(complete?"complete":"checkpointed")<<": "<<manifest["assets"].size()<<" assets\n";
    return complete?0:2;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
