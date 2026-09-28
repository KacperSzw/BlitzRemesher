#include "foliage.hpp"
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <archive.h>
#include <archive_entry.h>
#include <curl/curl.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>

namespace foliage {
namespace fs=std::filesystem;
constexpr uint64_t file_limit=1024ull*1024*1024,collection_limit=50ull*1024*1024*1024;
static std::string read(const fs::path& p) {
    if(fs::file_size(p)>file_limit)throw std::runtime_error("file exceeds 1 GiB: "+p.string());
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());
    return {std::istreambuf_iterator<char>(f),{}};
}
static Json json(const fs::path& p){return Json::parse(read(p));}
std::string sha256(std::string_view s) {
    unsigned char bytes[EVP_MAX_MD_SIZE];unsigned n{};
    if(EVP_Digest(s.data(),s.size(),bytes,&n,EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 failed");
    std::string out;for(unsigned i=0;i<n;++i){out+="0123456789abcdef"[bytes[i]>>4];out+="0123456789abcdef"[bytes[i]&15];}return out;
}
fs::path relative_path(const std::string& s) {
    fs::path p(s);
    if(s.empty()||s.find('\0')!=s.npos||s.find('\\')!=s.npos||s.find(':')!=s.npos||p.is_absolute())
        throw std::runtime_error("unsafe relative path: "+s);
    for(auto part:p)if(part=="..")throw std::runtime_error("parent traversal: "+s);
    p=p.lexically_normal();if(p.empty()||p==".")throw std::runtime_error("empty relative path");return p;
}
static fs::path destination(const fs::path& root,const std::string& relative) {
    auto out=root;for(auto part:relative_path(relative)) {
        out/=part;if(fs::is_symlink(fs::symlink_status(out)))throw std::runtime_error("symlink in destination: "+out.string());
    }
    return out;
}
static void write(const fs::path& p,std::string_view bytes) {
    if(!p.parent_path().empty())fs::create_directories(p.parent_path());
    auto tmp=p;tmp+=".part";
    if(fs::is_symlink(fs::symlink_status(tmp)))throw std::runtime_error("symlink staging file");
    {std::ofstream f(tmp,std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));if(!f)throw std::runtime_error("write failed: "+tmp.string());}
    fs::rename(tmp,p);
}
static void save(const fs::path& p,const Json& j){write(p,j.dump(2)+"\n");}
Json acquire_file(const fs::path& root,const Json& specification,const Fetch& fetch) {
    auto spec=specification;auto p=destination(root,spec.at("path"));
    const auto expected=spec.value("sha256",std::string());
    auto matches=[&](const std::string& bytes){return (expected.empty()||sha256(bytes)==expected)&&(!spec.contains("bytes")||bytes.size()==spec.at("bytes").get<uint64_t>());};
    std::string bytes;
    if(fs::exists(p)){bytes=read(p);if(!matches(bytes))bytes.clear();else if(!expected.empty())return spec;}
    // Unstamped local files are not trusted as downloads. A new acquisition
    // gets its bytes from the source; frozen replay reuses verified files.
    bytes=fetch(spec.at("url"));
    if(bytes.size()>file_limit||!matches(bytes))throw std::runtime_error("download size/hash mismatch: "+p.string());
    write(p,bytes);spec["bytes"]=bytes.size();spec["sha256"]=sha256(bytes);return spec;
}
Json extract_archive(const fs::path& input,const fs::path& root) {
    std::unique_ptr<archive,decltype(&archive_read_free)> a(archive_read_new(),archive_read_free);
    archive_read_support_filter_all(a.get());archive_read_support_format_all(a.get());
    if(archive_read_open_filename(a.get(),input.c_str(),65536)!=ARCHIVE_OK)throw std::runtime_error("cannot open archive");
    Json files=Json::array();archive_entry* entry{};uint64_t total=0;std::set<std::string> seen;
    int status;
    while((status=archive_read_next_header(a.get(),&entry))==ARCHIVE_OK) {
        const char* raw_name=archive_entry_pathname(entry);if(!raw_name)throw std::runtime_error("archive member has no name");
        std::string name=raw_name;auto path=relative_path(name);auto output=destination(root,path.generic_string());
        const auto type=archive_entry_filetype(entry);
        if(archive_entry_symlink(entry)||archive_entry_hardlink(entry)||(type!=AE_IFREG&&type!=AE_IFDIR))throw std::runtime_error("archive links/special files are unsupported");
        if(type==AE_IFDIR)continue;
        if(!seen.insert(path.generic_string()).second)throw std::runtime_error("duplicate archive member");
        const auto length=archive_entry_size(entry);
        if(length<0||uint64_t(length)>file_limit||uint64_t(length)+total>4*file_limit)throw std::runtime_error("expanded archive exceeds budget");
        total+=uint64_t(length);std::string bytes(size_t(length),'\0');size_t offset=0;
        while(offset<bytes.size()){auto n=archive_read_data(a.get(),bytes.data()+offset,bytes.size()-offset);if(n<=0)throw std::runtime_error("truncated archive member");offset+=size_t(n);}
        if(!fs::exists(output)||sha256(read(output))!=sha256(bytes))write(output,bytes);
        files.push_back({{"path",path.generic_string()},{"archive_member",name},{"bytes",bytes.size()},{"sha256",sha256(bytes)}});
    }
    if(status!=ARCHIVE_EOF)throw std::runtime_error("archive read failed");return files;
}
static std::chrono::steady_clock::time_point deadline;
static size_t receive(void* p,size_t a,size_t b,void* u) {
    auto& s=*static_cast<std::string*>(u);if(a&&b>file_limit/a)return 0;size_t n=a*b;
    if(s.size()+n>file_limit)return 0;try{s.append(static_cast<char*>(p),n);}catch(...){return 0;}return n;
}
static std::string request_http(const std::string& url,const std::string& post={}) {
    for(unsigned attempt=0;attempt<4;++attempt){
        auto remaining=std::chrono::duration_cast<std::chrono::seconds>(deadline-std::chrono::steady_clock::now()).count();
        if(remaining<=0)throw std::runtime_error("50-minute acquisition checkpoint; rerun the same command");
        std::unique_ptr<CURL,decltype(&curl_easy_cleanup)> c(curl_easy_init(),curl_easy_cleanup);
        if(!c)throw std::runtime_error("curl init");std::string bytes;
        curl_easy_setopt(c.get(),CURLOPT_URL,url.c_str());curl_easy_setopt(c.get(),CURLOPT_USERAGENT,"BlitzRemesher foliage research corpus");
        curl_easy_setopt(c.get(),CURLOPT_FOLLOWLOCATION,1L);curl_easy_setopt(c.get(),CURLOPT_PROTOCOLS_STR,"https,http");
        curl_easy_setopt(c.get(),CURLOPT_REDIR_PROTOCOLS_STR,"https,http");curl_easy_setopt(c.get(),CURLOPT_CONNECTTIMEOUT,20L);
        curl_easy_setopt(c.get(),CURLOPT_TIMEOUT,long(std::min<int64_t>(240,remaining)));curl_easy_setopt(c.get(),CURLOPT_MAXFILESIZE_LARGE,curl_off_t(file_limit));
        curl_easy_setopt(c.get(),CURLOPT_WRITEFUNCTION,receive);curl_easy_setopt(c.get(),CURLOPT_WRITEDATA,&bytes);
        if(!post.empty())curl_easy_setopt(c.get(),CURLOPT_POSTFIELDS,post.c_str());
        auto rc=curl_easy_perform(c.get());long code{};curl_easy_getinfo(c.get(),CURLINFO_RESPONSE_CODE,&code);
        if(rc==CURLE_OK&&code==200)return bytes;
        if(code==403||code==404)throw std::runtime_error("HTTP "+std::to_string(code)+": "+url);
        std::this_thread::sleep_for(std::chrono::seconds(1<<attempt));
    }
    throw std::runtime_error("download failed: "+url);
}
static std::string request(const std::string& url){return request_http(url);}
// Public, free itch.io uploads issue short-lived URLs. Store the stable upload
// endpoint and obtain a fresh URL on replay; never freeze a signed URL.
static std::string itch_download(const std::string& page,const std::string& endpoint) {
    auto html=request(page);const std::string marker="name=\"csrf_token\" value=\"";
    auto start=html.find(marker);if(start==html.npos)throw std::runtime_error("itch.io public download form changed");start+=marker.size();
    auto end=html.find('"',start);if(end==html.npos)throw std::runtime_error("invalid itch.io download form");
    std::unique_ptr<char,decltype(&curl_free)> escaped(curl_easy_escape(nullptr,html.data()+start,int(end-start)),curl_free);
    if(!escaped)throw std::runtime_error("URL encoding failed");
    auto response=Json::parse(request_http(endpoint,"csrf_token="+std::string(escaped.get())));
    return request(response.at("url"));
}
static std::string lower(std::string s){for(char& c:s)c=char(std::tolower(static_cast<unsigned char>(c)));return s;}
static Json acquire(const Json& selection,const fs::path& root) {
    fs::create_directories(root);auto checkpoint=root/"acquisition.json";
    Json result=fs::exists(checkpoint)?json(checkpoint):Json{{"version",1},{"packages",Json::array()}};
    result["selection_sha256"]=sha256(selection.dump());result["expected_packages"]=selection.at("packages").size();result["complete"]=false;
    for(const auto& source:selection.at("packages")) {
        const std::string id=source.at("id");relative_path(id);if(id.find('/')!=id.npos)throw std::runtime_error("invalid package id");
        auto found=std::find_if(result["packages"].begin(),result["packages"].end(),[&](const auto& p){return p.at("id")==id;});
        Json package=found==result["packages"].end()?source:*found;
        package["files"]=package.value("files",Json::array());package.erase("failure");package["complete"]=false;
        auto stash=[&]{if(found==result["packages"].end()){result["packages"].push_back(package);found=std::prev(result["packages"].end());}else *found=package;save(checkpoint,result);};
        auto download=[&](const std::string& relative,const std::string& url,std::string role="dependency") {
            const auto path="packages/"+id+"/"+relative;
            auto old=std::find_if(package["files"].begin(),package["files"].end(),[&](const auto& f){return f.at("path")==path;});
            Json spec=old==package["files"].end()?Json{{"path",path},{"url",url},{"role",role}}:*old;
            if(spec.value("url",std::string())!=url)throw std::runtime_error("source URL changed for frozen local file");
            Fetch fetch=request;
            if(source.at("provider")=="itch"&&role=="archive"){
                spec["transport"]="itch_upload";spec["page_url"]=source.at("source_url");
                fetch=[&](const std::string& endpoint){return itch_download(source.at("source_url"),endpoint);};
            }
            auto record=acquire_file(root,spec,fetch);
            if(old==package["files"].end())package["files"].push_back(record);else *old=record;
            stash();return root/path;
        };
        try {
            std::cout<<"Acquire "<<id<<std::endl;
            download("source-page.html",source.at("source_url"),"provenance");
            if(source.contains("preview_url"))download("provider-preview"+source.value("preview_extension",std::string(".jpg")),source.at("preview_url"),"provider_preview");
            if(source.at("provider")=="polyhaven") {
                const std::string asset=source.at("asset");
                auto files=json(download("files.json","https://api.polyhaven.com/files/"+asset,"provider_metadata"));
                auto info=json(download("info.json","https://api.polyhaven.com/info/"+asset,"provider_metadata"));
                package["authors"]=info.value("authors",Json::object());
                auto variants=files.at("gltf");const std::string resolution=variants.contains("2k")?"2k":"1k";
                auto entry=variants.at(resolution).at("gltf");package["resolution"]=resolution;
                download("source.gltf",entry.at("url"),"model");
                for(auto it=entry.at("include").begin();it!=entry.at("include").end();++it)download(it.key(),it.value().at("url"));
                package["opacity_maps"]=Json::array();
                for(auto it=files.begin();it!=files.end();++it)if(lower(it.key()).find("alpha")!=std::string::npos||lower(it.key()).find("opacity")!=std::string::npos) {
                    if(!it.value().contains(resolution)||!it.value().at(resolution).contains("png"))throw std::runtime_error("opacity map has no matching PNG variant");
                    auto map=it.value().at(resolution).at("png");std::string url=map.at("url");auto name=url.substr(url.find_last_of('/')+1);
                    auto p=download("opacity/"+name,url,"opacity");
                    package["opacity_maps"].push_back({{"provider_slot",it.key()},{"path",fs::relative(p,root).generic_string()},{"channel","R"}});
                }
            } else if(source.at("provider")=="archive"||source.at("provider")=="itch") {
                const std::string filename=source.value("archive_name",std::string("source.zip"));
                auto archive_path=download(filename,source.at("url"),"archive");
                auto members=extract_archive(archive_path,root/"packages"/id/"content");
                for(auto& f:members) {
                    f["path"]="packages/"+id+"/content/"+f.at("path").get<std::string>();f["archive_path"]=fs::relative(archive_path,root).generic_string();f["role"]="archive_member";
                    auto old=std::find_if(package["files"].begin(),package["files"].end(),[&](const auto& v){return v.at("path")==f.at("path");});
                    if(old==package["files"].end())package["files"].push_back(f);else if(old->at("sha256")!=f.at("sha256"))throw std::runtime_error("archive member changed");
                }
            } else throw std::runtime_error("unknown provider");
            package["complete"]=true;
        }catch(const std::exception& e){package["complete"]=false;package["failure"]=e.what();std::cerr<<id<<": "<<e.what()<<'\n';}
        stash();uint64_t total=0;for(auto& p:result["packages"])for(auto& f:p["files"])total+=f.at("bytes").get<uint64_t>();
        if(total>collection_limit)throw std::runtime_error("collection exceeds 50 GiB");
        if(std::chrono::steady_clock::now()>=deadline)break;
    }
    result["complete"]=std::all_of(selection.at("packages").begin(),selection.at("packages").end(),[&](const auto& s){return std::any_of(result["packages"].begin(),result["packages"].end(),[&](const auto& p){return p.at("id")==s.at("id")&&p.value("complete",false);});});
    save(checkpoint,result);return result;
}

// Collection-only inspection retains opacity metadata. It never produces a
// MeshView for the simplifier and never changes the opaque importer contract.
struct Point {double x{},y{},z{};auto operator<=>(const Point&)const=default;};
static Point operator-(Point a,Point b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static Point cross(Point a,Point b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static double dot(Point a,Point b){return a.x*b.x+a.y*b.y+a.z*b.z;}
struct Geometry {std::vector<Point> positions;std::vector<uint32_t> indices;bool uv{true};};
static Json geometry_stats(const Geometry& g) {
    if(g.positions.empty()||g.indices.empty()||g.indices.size()%3)throw std::runtime_error("empty/incomplete triangle mesh");
    if(g.positions.size()>UINT32_MAX||g.indices.size()/3>2000000)throw std::runtime_error("model exceeds corpus geometry budget");
    Point lo=g.positions[0],hi=lo;for(auto p:g.positions){if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("nonfinite vertex");lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};}
    std::vector<uint32_t> parents(g.positions.size());std::iota(parents.begin(),parents.end(),0);
    auto root=[&](uint32_t v){while(parents[v]!=v){parents[v]=parents[parents[v]];v=parents[v];}return v;};
    std::vector<uint64_t> edges;edges.reserve(g.indices.size());std::vector<std::array<Point,3>> canonical;canonical.reserve(g.indices.size()/3);
    uint64_t degenerate=0;
    for(size_t i=0;i<g.indices.size();i+=3){
        std::array<Point,3> triangle;
        for(unsigned j=0;j<3;++j){auto u=g.indices[i+j],v=g.indices[i+(j+1)%3];if(u>=g.positions.size()||v>=g.positions.size())throw std::runtime_error("index out of bounds");parents[root(v)]=root(u);triangle[j]=g.positions[u]-lo;if(u>v)std::swap(u,v);edges.push_back(uint64_t(u)<<32|v);}
        auto n=cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);degenerate+=dot(n,n)==0;
        std::sort(triangle.begin(),triangle.end());canonical.push_back(triangle);
    }
    struct Component{uint32_t faces{},boundary{};Point normal{};bool flat{true};};std::map<uint32_t,Component> components;
    for(size_t i=0;i<g.indices.size();i+=3){auto& c=components[root(g.indices[i])];auto n=cross(g.positions[g.indices[i+1]]-g.positions[g.indices[i]],g.positions[g.indices[i+2]]-g.positions[g.indices[i]]);double length=std::sqrt(dot(n,n));
        if(length){n={n.x/length,n.y/length,n.z/length};if(!c.faces)c.normal=n;else c.flat&=std::abs(dot(c.normal,n))>.997;}++c.faces;}
    std::sort(edges.begin(),edges.end());uint64_t boundaries=0,nonmanifold=0;
    for(size_t i=0;i<edges.size();){size_t j=i+1;while(j<edges.size()&&edges[j]==edges[i])++j;if(j-i==1){++boundaries;++components[root(uint32_t(edges[i]>>32))].boundary;}nonmanifold+=j-i>2;i=j;}
    uint32_t planar=0,open=0;for(auto& [_,c]:components){open+=c.boundary!=0;planar+=c.flat&&c.boundary>=3&&c.faces>=2;}
    std::sort(canonical.begin(),canonical.end());std::string fingerprint;fingerprint.reserve(canonical.size()*72);
    for(auto& t:canonical)for(auto p:t)for(double v:{p.x,p.y,p.z}){if(v==0)v=0;auto bits=std::bit_cast<uint64_t>(v);for(unsigned b=0;b<8;++b)fingerprint+=char(bits>>(b*8));}
    return {{"triangles",g.indices.size()/3},{"vertices",g.positions.size()},{"uv",g.uv},{"components",components.size()},{"open_components",open},
        {"planar_open_components",planar},{"boundary_edges",boundaries},{"nonmanifold_edges",nonmanifold},{"degenerate_triangles",degenerate},
        {"geometry_sha256",sha256(fingerprint)},{"hash_scope","sorted triangle positions; translation invariant; ignores winding, attributes and index order"},
        {"bounds",{{"min",{lo.x,lo.y,lo.z}},{"max",{hi.x,hi.y,hi.z}}}}};
}
static Json texture(const cgltf_texture_view& v,const cgltf_data& d) {
    if(!v.texture)return nullptr;Json t={{"texcoord",v.texcoord}};
    auto image=v.texture->image;if(image){t["image"]=size_t(image-d.images);if(image->uri)t["uri"]=image->uri;else t["embedded"]=true;}
    if(v.has_transform){const auto& x=v.transform;t["transform"]={{"offset",{x.offset[0],x.offset[1]}},{"scale",{x.scale[0],x.scale[1]}},{"rotation",x.rotation}};if(x.has_texcoord)t["texcoord"]=x.texcoord;}
    if(v.texture->sampler){auto& s=*v.texture->sampler;t["sampler"]={{"wrap_s",s.wrap_s},{"wrap_t",s.wrap_t},{"min_filter",s.min_filter},{"mag_filter",s.mag_filter}};}return t;
}
static std::vector<cgltf_node*> subtree(cgltf_node* root){
    std::vector<cgltf_node*> nodes{root};for(size_t i=0;i<nodes.size();++i)for(size_t n=0;n<nodes[i]->children_count;++n)nodes.push_back(nodes[i]->children[n]);return nodes;
}
static void append_mesh(const cgltf_data& d,const cgltf_mesh& mesh,Geometry& g,std::set<size_t>& materials,const float* transform=nullptr){
    for(size_t pi=0;pi<mesh.primitives_count;++pi){auto& p=mesh.primitives[pi];if(p.type!=cgltf_primitive_type_triangles||p.targets_count||p.has_draco_mesh_compression)throw std::runtime_error("unsupported primitive");
        const cgltf_accessor *positions=nullptr,*uv=nullptr;
        for(size_t a=0;a<p.attributes_count;++a){auto& t=p.attributes[a];if(t.type==cgltf_attribute_type_position)positions=t.data;if(t.type==cgltf_attribute_type_texcoord&&t.index==0)uv=t.data;}
        if(!positions||positions->count>UINT32_MAX-g.positions.size())throw std::runtime_error("invalid positions");const uint32_t base=uint32_t(g.positions.size());g.uv&=uv&&uv->count==positions->count;
        for(size_t i=0;i<positions->count;++i){float point[3];if(!cgltf_accessor_read_float(positions,i,point,3))throw std::runtime_error("position decode failed");
            if(transform){float out[3];for(unsigned axis=0;axis<3;++axis)out[axis]=transform[axis]*point[0]+transform[4+axis]*point[1]+transform[8+axis]*point[2]+transform[12+axis];std::copy(out,out+3,point);}
            g.positions.push_back({point[0],point[1],point[2]});if(uv){float t[2];if(!cgltf_accessor_read_float(uv,i,t,2)||!std::isfinite(t[0])||!std::isfinite(t[1]))throw std::runtime_error("invalid UV");}}
        const size_t count=p.indices?p.indices->count:positions->count;if(count%3||count/3>2000000||g.indices.size()/3+count/3>2000000)throw std::runtime_error("model exceeds two-million-triangle cap or incomplete primitive");
        for(size_t i=0;i<count;++i){auto index=p.indices?cgltf_accessor_read_index(p.indices,i):i;if(index>=positions->count)throw std::runtime_error("invalid index");g.indices.push_back(base+uint32_t(index));}
        if(p.material)materials.insert(size_t(p.material-d.materials));
    }
}
static Json gltf_inspect(const fs::path& path) {
    cgltf_options options{};cgltf_data* data{};
    if(cgltf_parse_file(&options,path.c_str(),&data)!=cgltf_result_success)throw std::runtime_error("glTF parse failed");
    std::unique_ptr<cgltf_data,decltype(&cgltf_free)> owner(data,cgltf_free);auto& d=*data;
    Json result={{"format","gltf"},{"models",Json::array()},{"materials",Json::array()},{"dependencies",Json::array()}};
    auto dependency=[&](const char* uri){if(uri&&!std::string_view(uri).starts_with("data:")){
        std::string decoded(uri);decoded.resize(cgltf_decode_uri(decoded.data()));
        auto p=relative_path(decoded);result["dependencies"].push_back(p.generic_string());
        if(!fs::is_regular_file(destination(path.parent_path(),p.generic_string())))throw std::runtime_error("missing dependency: "+p.string());}};
    for(size_t i=0;i<d.buffers_count;++i)dependency(d.buffers[i].uri);
    for(size_t i=0;i<d.images_count;++i)dependency(d.images[i].uri);
    if(d.skins_count||d.animations_count)throw std::runtime_error("animated/skinned asset");
    if(cgltf_load_buffers(&options,&d,path.c_str())!=cgltf_result_success||cgltf_validate(&d)!=cgltf_result_success)throw std::runtime_error("invalid glTF buffers");
    for(size_t i=0;i<d.materials_count;++i){auto& m=d.materials[i];result["materials"].push_back({{"index",i},{"name",m.name?m.name:""},
        {"alpha_mode",m.alpha_mode==cgltf_alpha_mode_mask?"MASK":m.alpha_mode==cgltf_alpha_mode_blend?"BLEND":"OPAQUE"},
        {"alpha_cutoff",m.alpha_cutoff},{"double_sided",bool(m.double_sided)},{"base_color_factor",{m.pbr_metallic_roughness.base_color_factor[0],m.pbr_metallic_roughness.base_color_factor[1],m.pbr_metallic_roughness.base_color_factor[2],m.pbr_metallic_roughness.base_color_factor[3]}},
        {"base_color",texture(m.pbr_metallic_roughness.base_color_texture,d)},{"normal",texture(m.normal_texture,d)},{"metallic_roughness",texture(m.pbr_metallic_roughness.metallic_roughness_texture,d)},{"occlusion",texture(m.occlusion_texture,d)},{"emissive",texture(m.emissive_texture,d)}});}
    for(size_t mi=0;mi<d.meshes_count;++mi){
        auto& mesh=d.meshes[mi];Geometry g;std::set<size_t> materials;
        try{
            append_mesh(d,mesh,g,materials);
            auto model=geometry_stats(g);model["selector"]={{"mesh",mi}};model["name"]=mesh.name?mesh.name:"";model["materials"]=materials;
            model["nodes"]=Json::array();for(size_t n=0;n<d.nodes_count;++n)if(d.nodes[n].mesh==&mesh){float transform[16];cgltf_node_transform_world(d.nodes+n,transform);model["nodes"].push_back({{"index",n},{"name",d.nodes[n].name?d.nodes[n].name:""},{"world_transform",std::vector<float>(transform,transform+16)}});}
            result["models"].push_back(model);
        }catch(const std::exception& e){result["models"].push_back({{"selector",{{"mesh",mi}}},{"name",mesh.name?mesh.name:""},{"failure",e.what()}});}
    }
    // A complete tree can be a trunk node with leaf/twig child meshes. Keep
    // this authored assembly as one candidate; never count its pieces as trees.
    for(size_t ni=0;ni<d.nodes_count;++ni){auto& node=d.nodes[ni];if(node.parent||!node.children_count)continue;auto nodes=subtree(&node);Geometry g;std::set<size_t> materials;Json members=Json::array();
        try{for(auto* n:nodes)if(n->mesh){float transform[16];cgltf_node_transform_world(n,transform);append_mesh(d,*n->mesh,g,materials,transform);members.push_back(size_t(n-d.nodes));}
            if(members.size()<2)continue;auto model=geometry_stats(g);model["selector"]={{"node",ni}};model["name"]=node.name?node.name:"";model["materials"]=materials;model["assembly_nodes"]=members;result["models"].push_back(model);
        }catch(const std::exception& e){result["models"].push_back({{"selector",{{"node",ni}}},{"name",node.name?node.name:""},{"failure",e.what()}});}
    }
    return result;
}
static Json obj_inspect(const fs::path& path,bool allow_missing_mtl) {
    tinyobj::ObjReader reader;tinyobj::ObjReaderConfig config;config.triangulate=true;config.mtl_search_path=path.parent_path().string();
    Json result={{"format","obj"},{"models",Json::array()},{"materials",Json::array()},{"dependencies",Json::array()},{"missing_material_libraries",Json::array()},{"source_material_names",Json::array()}};
    std::istringstream lines(read(path));std::string line;
    while(std::getline(lines,line)){
        std::istringstream fields(line);std::string directive;fields>>directive;
        if(directive=="usemtl"){std::string name;std::getline(fields>>std::ws,name);if(!name.empty()&&name.back()=='\r')name.pop_back();if(std::find(result["source_material_names"].begin(),result["source_material_names"].end(),Json(name))==result["source_material_names"].end())result["source_material_names"].push_back(name);}
        if(directive=="mtllib"){std::string name,extra;fields>>name;if(fields>>extra&&!extra.starts_with('#'))throw std::runtime_error("collection inspection requires one unescaped MTL filename per directive");auto p=relative_path(name);
            if(allow_missing_mtl&&!fs::exists(destination(path.parent_path(),p.generic_string())))result["missing_material_libraries"].push_back(p.generic_string());
            else result["dependencies"].push_back(p.generic_string());}
    }
    // Validate material paths before tinyobj opens their contents.
    for(auto& uri:result["dependencies"])if(!fs::is_regular_file(destination(path.parent_path(),uri.get<std::string>())))throw std::runtime_error("missing OBJ dependency: "+uri.get<std::string>());
    if(!reader.ParseFromFile(path.string(),config))throw std::runtime_error(reader.Error());result["warnings"]=reader.Warning();
    const auto& mats=reader.GetMaterials();
    for(size_t i=0;i<mats.size();++i){auto& m=mats[i];Json textures=Json::object();for(auto [slot,name]:std::initializer_list<std::pair<const char*,std::string>>{{"diffuse",m.diffuse_texname},{"alpha",m.alpha_texname},{"normal",m.normal_texname},{"bump",m.bump_texname},{"specular",m.specular_texname},{"ambient",m.ambient_texname},{"roughness",m.roughness_texname},{"metallic",m.metallic_texname},{"emissive",m.emissive_texname}})if(!name.empty()){textures[slot]=name;result["dependencies"].push_back(relative_path(name).generic_string());}
        result["materials"].push_back({{"index",i},{"name",m.name},{"dissolve",m.dissolve},{"alpha_mode",m.dissolve<1?"BLEND":m.alpha_texname.empty()?"UNSPECIFIED":"MAP_D"},{"textures",textures},{"sidedness","unspecified by OBJ/MTL"}});}
    for(auto& uri:result["dependencies"])if(!fs::is_regular_file(destination(path.parent_path(),uri.get<std::string>())))throw std::runtime_error("missing OBJ dependency: "+uri.get<std::string>());
    const auto& attr=reader.GetAttrib();
    for(size_t si=0;si<reader.GetShapes().size();++si){const auto& shape=reader.GetShapes()[si];Geometry g;std::set<int> materials;
        std::map<int,uint32_t> vertices;
        for(auto index:shape.mesh.indices){if(index.vertex_index<0||size_t(index.vertex_index)*3+2>=attr.vertices.size())throw std::runtime_error("invalid OBJ vertex");auto [it,added]=vertices.try_emplace(index.vertex_index,uint32_t(g.positions.size()));if(added){size_t n=size_t(index.vertex_index)*3;g.positions.push_back({attr.vertices[n],attr.vertices[n+1],attr.vertices[n+2]});}g.indices.push_back(it->second);g.uv&=index.texcoord_index>=0&&size_t(index.texcoord_index)*2+1<attr.texcoords.size();}
        for(int m:shape.mesh.material_ids)if(m>=0)materials.insert(m);
        auto model=geometry_stats(g);model["selector"]={{"shape",si}};model["name"]=shape.name;model["materials"]=materials;result["models"].push_back(model);
    }
    return result;
}
Json inspect(const fs::path& path,bool allow_missing_mtl) {
    if(fs::file_size(path)>file_limit)throw std::runtime_error("model file too large");auto ext=lower(path.extension().string());
    if(ext==".gltf"||ext==".glb")return gltf_inspect(path);if(ext==".obj")return obj_inspect(path,allow_missing_mtl);throw std::runtime_error("unsupported collection model format");
}
static const Json& selected_model(const Json& inspection,const Json& selector) {
    for(auto& m:inspection.at("models"))if(m.at("selector")==selector){if(m.contains("failure"))throw std::runtime_error(m.at("failure").get<std::string>());return m;}
    throw std::runtime_error("model selector is missing");
}
Json check(const Json& manifest,const fs::path& root) {
    Json report={{"version",1},{"manifest_sha256",sha256(manifest.dump())},{"complete",false},{"failures",Json::array()},{"assets",Json::array()}};
    if(manifest.at("kind")!="foliage_collection"||manifest.at("benchmark_eligible")!=false)throw std::runtime_error("not an unscored foliage collection");
    std::map<std::string,Json> files,packages,inspections;uint64_t total=0;std::set<std::string> ids,identities,geometry;
    for(auto& p:manifest.at("packages")){
        std::string id=p.at("id");if(!packages.emplace(id,p).second)throw std::runtime_error("duplicate package id");
        if(p.at("license")!="CC0-1.0"||!p.at("complete").get<bool>())throw std::runtime_error("unverified package");
        for(auto& f:p.at("files")){
            std::string name=f.at("path");if(!files.emplace(name,f).second)throw std::runtime_error("duplicate file record");
            total+=f.at("bytes").get<uint64_t>();if(total>collection_limit)throw std::runtime_error("collection exceeds byte cap");
            try{auto data=read(destination(root,name));if(data.size()!=f.at("bytes").get<uint64_t>()||sha256(data)!=f.at("sha256").get<std::string>())throw std::runtime_error("size/hash mismatch");}
            catch(const std::exception& e){report["failures"].push_back({{"file",name},{"error",e.what()}});}
        }
    }
    std::map<std::string,unsigned> counts;
    for(auto& asset:manifest.at("assets")){
        const std::string id=asset.at("id");Json checked={{"id",id},{"passed",false}};
        try{
            if(!ids.insert(id).second||!identities.insert(asset.at("authored_identity")).second)throw std::runtime_error("duplicate model identity");
            if(asset.at("split")!="unassigned"||asset.at("bake_status")!="not_baked"||!asset.at("bake_seconds").is_null()||asset.at("vertex_mode")!="original_source")throw std::runtime_error("collection assets must remain unscored and unbaked");
            const std::string package_id=asset.at("package");if(!packages.contains(package_id))throw std::runtime_error("missing package");
            const std::string path=asset.at("model");if(!files.contains(path))throw std::runtime_error("unregistered model file");
            const auto key=path+(asset.value("allow_missing_mtl",false)?"|missing-mtl":"");
            if(!inspections.contains(key))inspections[key]=inspect(destination(root,path),asset.value("allow_missing_mtl",false));auto& inspected=inspections.at(key);
            auto& model=selected_model(inspected,asset.at("selector"));
            if(model!=asset.at("geometry"))throw std::runtime_error("model geometry inspection changed");
            if(!geometry.insert(model.at("geometry_sha256")).second)throw std::runtime_error("duplicate triangle geometry");
            if(!model.at("uv").get<bool>()||model.at("open_components")==0)throw std::runtime_error("expected UV-mapped open card surfaces");
            if(asset.at("card_evidence").get<std::string>().empty()||asset.at("source_group").get<std::string>().empty())throw std::runtime_error("missing card/family review");
            for(auto& dep:inspected.at("dependencies"))if(!files.contains((fs::path(path).parent_path()/relative_path(dep)).lexically_normal().generic_string()))throw std::runtime_error("unregistered model dependency");
            if(asset.at("materials")!=inspected.at("materials"))throw std::runtime_error("material metadata changed");
            if(asset.value("allow_missing_mtl",false)&&(!asset.at("requires_material_setup").get<bool>()||asset.at("missing_material_libraries")!=inspected.at("missing_material_libraries")))throw std::runtime_error("missing material library must be disclosed");
            if(asset.at("opacity_bindings").empty())throw std::runtime_error("missing opacity mapping");
            for(auto& binding:asset.at("opacity_bindings")){
                const bool unassigned=binding.at("material").is_null();
                const size_t mat=unassigned?0:binding.at("material").get<size_t>();
                if(unassigned){if(!model.at("materials").empty()||!asset.at("requires_material_setup").get<bool>()||binding.at("scope")!="whole_model"||!binding.contains("path"))throw std::runtime_error("unassigned material requires an explicit whole-model mapping");}
                else if(mat>=inspected.at("materials").size()||std::find(model.at("materials").begin(),model.at("materials").end(),Json(mat))==model.at("materials").end())throw std::runtime_error("opacity material is not used by the selected model");
                if(binding.at("channel")!="A"&&binding.at("channel")!="R")throw std::runtime_error("unknown opacity channel");
                if(binding.contains("path")){if(!files.contains(binding.at("path")))throw std::runtime_error("missing opacity file record");}
                else if(binding.contains("image")){auto& base=inspected.at("materials").at(mat).at("base_color");if(base.is_null()||base.at("image")!=binding.at("image")||binding.at("channel")!="A")throw std::runtime_error("embedded/base-color opacity mapping mismatch");}
                else throw std::runtime_error("unresolved opacity mapping");
            }
            ++counts[asset.at("category")];checked["passed"]=true;
        }catch(const std::exception& e){checked["error"]=e.what();report["failures"].push_back({{"asset",id},{"error",e.what()}});}
        report["assets"].push_back(checked);
    }
    for(auto it=manifest.at("targets").begin();it!=manifest.at("targets").end();++it)if(counts[it.key()]!=it.value().get<unsigned>())report["failures"].push_back({{"category",it.key()},{"expected",it.value()},{"actual",counts[it.key()]}});
    unsigned expected=0;for(auto& n:manifest.at("targets"))expected+=n.get<unsigned>();
    if(ids.size()!=expected)report["failures"].push_back({{"error","asset count does not match target total"}});
    report["counts"]=counts;report["files"]=files.size();report["bytes"]=total;report["complete"]=report["failures"].empty();return report;
}
static Json freeze(const Json& selection,const fs::path& root,const fs::path& snapshots) {
    const auto acquisition=json(root/"acquisition.json");auto original=json("research/corpus.json");
    Json manifest=selection;manifest["version"]=1;manifest["kind"]="foliage_collection";manifest["benchmark_eligible"]=false;
    manifest["original_corpus_sha256"]=sha256(read("research/corpus.json"));manifest["packages"]=Json::array();
    std::set<std::string> wanted;std::map<std::string,Json> inspected;
    for(auto& asset:manifest["assets"]){
        wanted.insert(asset.at("package"));const std::string path=asset.at("model");if(!inspected.contains(path))inspected[path]=inspect(destination(root,path),asset.value("allow_missing_mtl",false));
        asset["geometry"]=selected_model(inspected[path],asset.at("selector"));asset["materials"]=inspected[path].at("materials");
        asset["split"]="unassigned";asset["bake_status"]="not_baked";asset["bake_seconds"]=nullptr;asset["vertex_mode"]="original_source";
        if(asset.value("allow_missing_mtl",false))asset["missing_material_libraries"]=inspected[path].at("missing_material_libraries");
    }
    for(auto& p:acquisition.at("packages"))if(wanted.contains(p.at("id"))){
        if(p.at("provider")=="polyhaven")for(auto& a:original.at("assets"))if(a.at("provider")=="polyhaven"&&a.at("source_identity")==p.at("asset"))throw std::runtime_error("source already represented in original corpus");
        auto package=p;
        for(auto& f:package["files"])if(f.at("role")=="provenance"||f.at("role")=="provider_metadata"){
            auto snapshot=snapshots/p.at("id").get<std::string>()/fs::path(f.at("path").get<std::string>()).filename();
            write(snapshot,read(destination(root,f.at("path"))));f["transport"]="snapshot";f["snapshot"]=fs::relative(snapshot).generic_string();
        }
        manifest["packages"].push_back(package);
    }
    return manifest;
}
static void replay(const Json& manifest,const fs::path& root) {
    for(auto& package:manifest.at("packages")){
        for(auto& f:package.at("files"))if(f.contains("url")){
            Fetch fetch=request;
            if(f.value("transport",std::string())=="itch_upload")fetch=[&](const std::string& url){return itch_download(f.at("page_url"),url);};
            if(f.value("transport",std::string())=="snapshot")fetch=[&](const std::string&){return read(destination(".",f.at("snapshot")));};
            acquire_file(root,f,fetch);
        }
        std::set<std::string> archives;
        for(auto& f:package.at("files"))if(f.contains("archive_path")){
            auto p=destination(root,f.at("path"));if(fs::exists(p)&&sha256(read(p))==f.at("sha256").get<std::string>())continue;
            const std::string name=f.at("archive_path");if(archives.insert(name).second)extract_archive(destination(root,name),root/"packages"/package.at("id").get<std::string>()/"content");
        }
        std::cout<<package.at("id")<<" replayed\n";
    }
}
static std::string base64(std::string_view bytes){
    std::string out(4*((bytes.size()+2)/3)+1,'\0');auto n=EVP_EncodeBlock(reinterpret_cast<unsigned char*>(out.data()),reinterpret_cast<const unsigned char*>(bytes.data()),int(bytes.size()));out.resize(size_t(n));return out;
}
static void word(std::string& bytes,uint32_t v){for(unsigned s=0;s<32;s+=8)bytes+=char(v>>s);}
// Presentation streams are disposable derivatives, never import replacements.
// Alpha is displayed from the sidecar; no simplification or scoring runs here.
static Json previews(const Json& manifest,const fs::path& root){
    auto verified=check(manifest,root);if(!verified.at("complete").get<bool>())throw std::runtime_error("preview requires a verified collection");
    Json out={{"manifest_sha256",sha256(manifest.dump())},{"images",Json::object()},{"assets",Json::array()}};
    auto image=[&](const std::string& bytes){auto hash=sha256(bytes);if(!out["images"].contains(hash)){const char* mime=bytes.starts_with("\x89PNG")?"image/png":"image/jpeg";out["images"][hash]="data:"+std::string(mime)+";base64,"+base64(bytes);}return hash;};
    for(auto& asset:manifest.at("assets")){
        const fs::path path=destination(root,asset.at("model"));Json output={{"id",asset.at("id")},{"triangles",asset.at("geometry").at("triangles")},{"preview",asset.at("preview")},{"primitives",Json::array()}};
        auto binding=[&](const Json& material)->Json{for(auto& b:asset.at("opacity_bindings"))if(b.at("material")==material)return b;return nullptr;};
        auto primitive=[&](const std::string& vertices,const std::string& indices,const std::string& color,const Json& b){
            Json p={{"vertices",base64(vertices)},{"indices",base64(indices)},{"color",color},{"opacity",color},{"channel",0},{"cutoff",0.0},{"factor",{1,1,1,1}}};
            if(!b.is_null()){if(b.contains("path"))p["opacity"]=image(read(destination(root,b.at("path"))));p["channel"]=b.at("channel")=="R"?1:0;p["cutoff"]=b.at("preview_cutoff");}return p;
        };
        if(asset.at("selector").contains("shape")){
            tinyobj::ObjReader reader;tinyobj::ObjReaderConfig config;config.triangulate=true;config.mtl_search_path=path.parent_path().string();if(!reader.ParseFromFile(path.string(),config))throw std::runtime_error("OBJ preview parse");
            auto& a=reader.GetAttrib();auto& shape=reader.GetShapes().at(asset.at("selector").at("shape"));std::string vertices,indices;
            for(auto i:shape.mesh.indices){for(unsigned n=0;n<3;++n)word(vertices,std::bit_cast<uint32_t>(float(a.vertices[size_t(i.vertex_index)*3+n])));for(unsigned n=0;n<2;++n)word(vertices,std::bit_cast<uint32_t>(float(a.texcoords[size_t(i.texcoord_index)*2+n])));word(indices,uint32_t(indices.size()/4));}
            auto b=binding(nullptr);auto color=image(read(destination(root,b.at("base_color_path"))));output["primitives"].push_back(primitive(vertices,indices,color,b));
        }else{
            cgltf_options options{};cgltf_data* raw{};if(cgltf_parse_file(&options,path.c_str(),&raw)!=cgltf_result_success)throw std::runtime_error("glTF preview parse");std::unique_ptr<cgltf_data,decltype(&cgltf_free)> owner(raw,cgltf_free);auto& d=*raw;
            if(cgltf_load_buffers(&options,&d,path.c_str())!=cgltf_result_success)throw std::runtime_error("glTF preview buffers");
            auto texture_image=[&](const cgltf_image* im){if(!im)throw std::runtime_error("missing preview color image");if(im->buffer_view){auto v=im->buffer_view;return image(std::string(reinterpret_cast<const char*>(cgltf_buffer_view_data(v)),v->size));}
                if(!im->uri||std::string_view(im->uri).starts_with("data:"))throw std::runtime_error("unsupported preview image URI");std::string uri(im->uri);uri.resize(cgltf_decode_uri(uri.data()));return image(read(destination(path.parent_path(),uri)));};
            auto mesh=[&](cgltf_mesh* m,const float* transform){for(size_t pi=0;pi<m->primitives_count;++pi){auto& p=m->primitives[pi];const cgltf_accessor *pos=nullptr,*uv=nullptr;
                for(size_t i=0;i<p.attributes_count;++i){auto& a=p.attributes[i];if(a.type==cgltf_attribute_type_position)pos=a.data;if(a.type==cgltf_attribute_type_texcoord&&a.index==0)uv=a.data;}
                if(!pos||!uv||!p.material||!p.material->pbr_metallic_roughness.base_color_texture.texture)throw std::runtime_error("preview needs positions, UV0 and base color");
                auto& color_view=p.material->pbr_metallic_roughness.base_color_texture;std::string vertices,indices;
                for(size_t i=0;i<pos->count;++i){float point[3],t[2];if(!cgltf_accessor_read_float(pos,i,point,3)||!cgltf_accessor_read_float(uv,i,t,2))throw std::runtime_error("preview vertex decode");
                    for(unsigned axis=0;axis<3;++axis){float v=point[axis];if(transform)v=transform[axis]*point[0]+transform[4+axis]*point[1]+transform[8+axis]*point[2]+transform[12+axis];word(vertices,std::bit_cast<uint32_t>(v));}
                    if(color_view.has_transform){auto& x=color_view.transform;float u=t[0]*x.scale[0],v=t[1]*x.scale[1];t[0]=std::cos(x.rotation)*u-std::sin(x.rotation)*v+x.offset[0];t[1]=std::sin(x.rotation)*u+std::cos(x.rotation)*v+x.offset[1];}
                    for(float v:t)word(vertices,std::bit_cast<uint32_t>(v));}
                size_t count=p.indices?p.indices->count:pos->count;for(size_t i=0;i<count;++i)word(indices,uint32_t(p.indices?cgltf_accessor_read_index(p.indices,i):i));
                auto item=primitive(vertices,indices,texture_image(color_view.texture->image),binding(size_t(p.material-d.materials)));
                item["material"]=size_t(p.material-d.materials);
                auto* factor=p.material->pbr_metallic_roughness.base_color_factor;item["factor"]={factor[0],factor[1],factor[2],factor[3]};output["primitives"].push_back(item);
            }};
            if(asset.at("selector").contains("node")){auto n=asset.at("selector").at("node").get<size_t>();for(auto* node:subtree(d.nodes+n))if(node->mesh){float transform[16];cgltf_node_transform_world(node,transform);mesh(node->mesh,transform);}}
            else {const auto mi=asset.at("selector").at("mesh").get<size_t>();float transform[16];const float* matrix=nullptr;for(size_t n=0;n<d.nodes_count;++n)if(d.nodes[n].mesh==d.meshes+mi){cgltf_node_transform_world(d.nodes+n,transform);matrix=transform;break;}mesh(d.meshes+mi,matrix);}
        }
        out["assets"].push_back(output);
    }
    return out;
}
static std::string unbase64(const std::string& encoded){
    std::string bytes(encoded.size()/4*3,'\0');auto n=EVP_DecodeBlock(reinterpret_cast<unsigned char*>(bytes.data()),reinterpret_cast<const unsigned char*>(encoded.data()),int(encoded.size()));
    if(n<0)throw std::runtime_error("invalid presentation buffer");if(encoded.ends_with("="))--n;if(encoded.ends_with("=="))--n;bytes.resize(size_t(n));return bytes;
}
// Explicit geometry-only experiment inputs. Originals remain intact and the
// collection stays benchmark-ineligible. Material alpha is deliberately absent
// from these disposable files, so no resulting audit is an opacity audit.
static void geometry_inputs(const Json& manifest,const fs::path& root,const fs::path& output,const std::set<std::string>& selected){
    auto data=previews(manifest,root);Json records={{"version",1},{"scope","foliage_card_geometry_only"},{"benchmark_eligible",false},{"manifest_sha256",data.at("manifest_sha256")},{"assets",Json::array()}};
    for(auto& source:data.at("assets")){
        const std::string id=source.at("id");if(!selected.empty()&&!selected.contains(id))continue;auto dir=output/relative_path(id);fs::create_directories(dir);std::string binary;
        Json gltf={{"asset",{{"version","2.0"},{"generator","BlitzRemesher geometry-only foliage experiment"}}},{"extras",{{"scope","opaque card geometry; no texture/opacity/normal/color audit"}}},
            {"scene",0},{"scenes",Json::array({{{"nodes",{0}}}})},{"nodes",Json::array({{{"mesh",0},{"name",id}}})},{"materials",Json::array()},
            {"bufferViews",Json::array()},{"accessors",Json::array()},{"meshes",Json::array({{{"primitives",Json::array()}}})}};
        auto asset=std::find_if(manifest.at("assets").begin(),manifest.at("assets").end(),[&](const auto& a){return a.at("id")==id;});
        for(auto& m:asset->at("materials"))gltf["materials"].push_back({{"name",m.value("name",std::string("card"))},{"doubleSided",m.value("double_sided",true)},{"alphaMode","OPAQUE"}});
        if(gltf["materials"].empty())gltf["materials"].push_back({{"name","card"},{"doubleSided",true},{"alphaMode","OPAQUE"}});
        for(auto& p:source.at("primitives")){
            auto vertices=unbase64(p.at("vertices")),indices=unbase64(p.at("indices"));const size_t view=gltf["bufferViews"].size(),accessor=gltf["accessors"].size();
            gltf["bufferViews"].push_back({{"buffer",0},{"byteOffset",binary.size()},{"byteLength",vertices.size()},{"byteStride",20}});binary+=vertices;
            gltf["bufferViews"].push_back({{"buffer",0},{"byteOffset",binary.size()},{"byteLength",indices.size()}});binary+=indices;
            gltf["accessors"].push_back({{"bufferView",view},{"componentType",5126},{"count",vertices.size()/20},{"type","VEC3"}});
            gltf["accessors"].push_back({{"bufferView",view},{"byteOffset",12},{"componentType",5126},{"count",vertices.size()/20},{"type","VEC2"}});
            gltf["accessors"].push_back({{"bufferView",view+1},{"componentType",5125},{"count",indices.size()/4},{"type","SCALAR"}});
            gltf["meshes"][0]["primitives"].push_back({{"attributes",{{"POSITION",accessor},{"TEXCOORD_0",accessor+1}}},{"indices",accessor+2},{"material",p.value("material",0)}});
        }
        gltf["buffers"]=Json::array({{{"uri","source.bin"},{"byteLength",binary.size()}}});write(dir/"source.bin",binary);save(dir/"source.gltf",gltf);
        records["assets"].push_back({{"id",id},{"triangles",source.at("triangles")},{"gltf_sha256",sha256(read(dir/"source.gltf"))},{"binary_sha256",sha256(binary)},
            {"source_model",asset->at("model")},{"source_selector",asset->at("selector")},{"source_geometry_sha256",asset->at("geometry").at("geometry_sha256")},
            {"vertex_contract","World-transformed imported positions and UV0; reuse refers to this immutable LOD0 buffer, not original file bytes"}});
    }
    if(!selected.empty()&&records["assets"].size()!=selected.size())throw std::runtime_error("missing selected geometry input");save(output/"manifest.json",records);
}
int main(int argc,char** argv) {
    if(argc<2)throw std::invalid_argument("foliage-acquire SOURCES ROOT | foliage-inventory ROOT OUTPUT | foliage-inspect MODEL OUTPUT | foliage-freeze SELECTION ROOT MANIFEST | foliage-check MANIFEST ROOT REPORT | foliage-replay MANIFEST ROOT | foliage-previews MANIFEST ROOT OUTPUT");
    curl_global_init(CURL_GLOBAL_DEFAULT);deadline=std::chrono::steady_clock::now()+std::chrono::minutes(50);
    std::string command=argv[0];
    if(command=="foliage-acquire"&&argc==3){auto r=acquire(json(argv[1]),argv[2]);return r.at("complete").get<bool>()?0:2;}
    if(command=="foliage-freeze"&&argc==4){auto m=freeze(json(argv[1]),argv[2],fs::path(argv[3]).parent_path()/"provenance");auto report=check(m,argv[2]);save(fs::path(argv[3]).string()+".check.json",report);if(report.at("complete")!=true)throw std::runtime_error("freeze verification failed; see check report");save(argv[3],m);return 0;}
    if(command=="foliage-check"&&argc==4){auto r=check(json(argv[1]),argv[2]);save(argv[3],r);std::cout<<r.at("counts").dump()<<'\n';return r.at("complete").get<bool>()?0:2;}
    if(command=="foliage-replay"&&argc==3){auto m=json(argv[1]);replay(m,argv[2]);auto r=check(m,argv[2]);save(fs::path(argv[2])/"replay-check.json",r);return r.at("complete").get<bool>()?0:2;}
    if(command=="foliage-previews"&&argc==4){save(argv[3],previews(json(argv[1]),argv[2]));return 0;}
    if(command=="foliage-geometry"&&argc>=4){std::set<std::string> selected;for(int i=4;i<argc;++i)selected.insert(argv[i]);geometry_inputs(json(argv[1]),argv[2],argv[3],selected);return 0;}
    if(command=="foliage-inspect"&&argc==3){save(argv[2],inspect(argv[1]));return 0;}
    if(command=="foliage-inventory"&&argc==3){fs::path root=argv[1];auto acquisition=json(root/"acquisition.json");Json inventory={{"packages",Json::array()}};
        for(auto& p:acquisition["packages"]){Json package={{"id",p.at("id")},{"models",Json::array()}};for(auto& f:p["files"]){fs::path path=f.at("path").get<std::string>();auto ext=lower(path.extension().string());if(ext!=".gltf"&&ext!=".glb"&&ext!=".obj")continue;
            try{auto entry=inspect(root/path,true);entry["path"]=path.generic_string();package["models"].push_back(entry);}catch(const std::exception& e){package["models"].push_back({{"path",path.generic_string()},{"failure",e.what()}});}}
            inventory["packages"].push_back(package);std::cout<<p.at("id")<<" inspected\n";save(argv[2],inventory);}
        return 0;}
    throw std::invalid_argument("invalid foliage command or arguments");
}
}
