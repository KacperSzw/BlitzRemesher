#include "../tools/foliage.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <bit>
namespace fs=std::filesystem;
using foliage::Json;
#define CHECK(x) do {if(!(x))throw std::runtime_error("foliage contract: " #x);}while(0)
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}CHECK(rejected);}
static void write(const fs::path& p,const std::string& bytes){fs::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);f<<bytes;}
static void archive_fixture(const fs::path& path,const std::string& name,bool symlink=false){
    auto* a=archive_write_new();archive_write_set_format_pax_restricted(a);CHECK(archive_write_open_filename(a,path.c_str())==ARCHIVE_OK);
    auto* e=archive_entry_new();archive_entry_set_pathname(e,name.c_str());archive_entry_set_perm(e,0644);
    archive_entry_set_filetype(e,symlink?AE_IFLNK:AE_IFREG);archive_entry_set_size(e,symlink?0:4);
    if(symlink)archive_entry_set_symlink(e,"../outside");CHECK(archive_write_header(a,e)==ARCHIVE_OK);
    if(!symlink)CHECK(archive_write_data(a,"mesh",4)==4);archive_entry_free(e);archive_write_close(a);archive_write_free(a);
}
int main(){auto root=fs::temp_directory_path()/("blitz-foliage-contracts-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));try{
    fs::create_directories(root);
    for(auto p:{"../escape","/absolute","C:/drive","a/../../escape","a\\b",""})rejects([&]{foliage::relative_path(p);});
    CHECK(foliage::relative_path("textures/leaf.png")=="textures/leaf.png");
    CHECK(foliage::sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    Json spec={{"path","nested/model"},{"url","fixture"},{"bytes",4},{"sha256",foliage::sha256("mesh")}};unsigned calls=0;
    auto fetch=[&](const std::string&){++calls;return std::string("mesh");};
    foliage::acquire_file(root,spec,fetch);foliage::acquire_file(root,spec,fetch);CHECK(calls==1);
    write(root/"nested/model","bad");foliage::acquire_file(root,spec,fetch);CHECK(calls==2);
    rejects([&]{foliage::acquire_file(root,Json{{"path","bad"},{"url","fixture"},{"sha256",foliage::sha256("other")}},fetch);});CHECK(!fs::exists(root/"bad"));
    rejects([&]{foliage::acquire_file(root,Json{{"path","interrupted"},{"url","fixture"}},[](const std::string&)->std::string{throw std::runtime_error("disconnect");});});CHECK(!fs::exists(root/"interrupted"));
    archive_fixture(root/"good.tar","nested/mesh.obj");auto files=foliage::extract_archive(root/"good.tar",root/"expanded");CHECK(files.size()==1&&files[0]["sha256"]==foliage::sha256("mesh"));
    archive_fixture(root/"bad.tar","../escape");rejects([&]{foliage::extract_archive(root/"bad.tar",root/"expanded");});CHECK(!fs::exists(root/"escape"));
    archive_fixture(root/"link.tar","link",true);rejects([&]{foliage::extract_archive(root/"link.tar",root/"expanded");});
    const std::string obj="mtllib card.mtl\no card\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl leaf\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n";
    write(root/"card.obj",obj);write(root/"card.mtl","newmtl leaf\nmap_d alpha.png\n");
    rejects([&]{foliage::inspect(root/"card.obj");});write(root/"alpha.png","fixture");
    auto report=foliage::inspect(root/"card.obj");CHECK(report["models"][0]["triangles"]==2&&report["models"][0]["planar_open_components"]==1);CHECK(report["materials"][0]["alpha_mode"]=="MAP_D");
    Json records=Json::array();for(const auto* name:{"card.obj","card.mtl","alpha.png"}){std::ifstream f(root/name,std::ios::binary);std::string bytes{std::istreambuf_iterator<char>(f),{}};records.push_back({{"path",name},{"sha256",foliage::sha256(bytes)},{"bytes",bytes.size()}});}
    Json asset={{"id","card"},{"package","fixture"},{"model","card.obj"},{"selector",{{"shape",0}}},{"geometry",report["models"][0]},
        {"materials",report["materials"]},{"authored_identity","fixture:card"},{"source_group","fixture"},{"card_evidence","Test-owned two-triangle UV card"},
        {"split","unassigned"},{"bake_status","not_baked"},{"bake_seconds",nullptr},{"vertex_mode","original_source"},{"category","grass"},{"opacity_bindings",Json::array({{{"material",0},{"path","alpha.png"},{"channel","R"}}})}};
    Json manifest={{"kind","foliage_collection"},{"benchmark_eligible",false},{"targets",{{"grass",1}}},{"packages",Json::array({{{"id","fixture"},{"license","CC0-1.0"},{"complete",true},{"files",records}}})},{"assets",Json::array({asset})}};
    CHECK(foliage::check(manifest,root)["complete"]==true);
    auto invalid=manifest;invalid["assets"][0]["opacity_bindings"][0]["material"]=1;CHECK(foliage::check(invalid,root)["complete"]==false);
    invalid=manifest;invalid["targets"]["grass"]=2;CHECK(foliage::check(invalid,root)["complete"]==false);
    invalid=manifest;invalid["targets"]["grass"]=2;auto duplicate=asset;duplicate["id"]="renamed";duplicate["authored_identity"]="new-label-same-geometry";invalid["assets"].push_back(duplicate);CHECK(foliage::check(invalid,root)["complete"]==false);
    invalid=manifest;invalid["benchmark_eligible"]=true;rejects([&]{foliage::check(invalid,root);});
    invalid=manifest;invalid["assets"][0]["bake_seconds"]=0;CHECK(foliage::check(invalid,root)["complete"]==false);
    write(root/"alpha.png","changed");CHECK(foliage::check(manifest,root)["complete"]==false);write(root/"alpha.png","fixture");
    fs::remove(root/"card.mtl");rejects([&]{foliage::inspect(root/"card.obj");});auto missing=foliage::inspect(root/"card.obj",true);CHECK(missing["missing_material_libraries"]==Json::array({"card.mtl"}));CHECK(missing["materials"].empty());
    write(root/"unsafe.obj"," \tmtllib\t../outside.mtl\n"+obj);rejects([&]{foliage::inspect(root/"unsafe.obj",true);});

    std::string binary;auto word=[&](uint32_t v){for(unsigned i=0;i<4;i++)binary+=char(v>>(8*i));};
    for(float v:{0.f,0.f,0.f,1.f,0.f,0.f,1.f,1.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f,1.f,1.f,0.f,1.f})word(std::bit_cast<uint32_t>(v));
    for(uint16_t i:{0,1,2,0,2,3}){binary+=char(i);binary+=char(i>>8);}write(root/"quad.bin",binary);
    Json gltf={{"asset",{{"version","2.0"}}},{"buffers",Json::array({{{"uri","quad.bin"},{"byteLength",binary.size()}}})},
      {"bufferViews",Json::array({{{"buffer",0},{"byteLength",48}},{{"buffer",0},{"byteOffset",48},{"byteLength",32}},{{"buffer",0},{"byteOffset",80},{"byteLength",12}}})},
      {"accessors",Json::array({{{"bufferView",0},{"componentType",5126},{"count",4},{"type","VEC3"}},{{"bufferView",1},{"componentType",5126},{"count",4},{"type","VEC2"}},{{"bufferView",2},{"componentType",5123},{"count",6},{"type","SCALAR"}}})},
      {"images",Json::array({{{"uri","alpha.png"}}})},{"textures",Json::array({{{"source",0}}})},
      {"materials",Json::array({{{"alphaMode","MASK"},{"alphaCutoff",0.37},{"doubleSided",false},{"pbrMetallicRoughness",{{"baseColorTexture",{{"index",0},{"texCoord",0}}}}}}})},
      {"meshes",Json::array({{{"primitives",Json::array({{{"attributes",{{"POSITION",0},{"TEXCOORD_0",1}}},{"indices",2},{"material",0}}})}}})},
      {"nodes",Json::array({{{"children",{1,2}},{"translation",{5,0,0}}},{{"mesh",0}},{{"mesh",0},{"translation",{0,2,0}}}})}};
    write(root/"quad.gltf",gltf.dump());auto inspected=foliage::inspect(root/"quad.gltf");
    CHECK(inspected["materials"][0]["alpha_mode"]=="MASK");CHECK(std::abs(inspected["materials"][0]["alpha_cutoff"].get<double>()-.37)<1e-6);CHECK(inspected["materials"][0]["double_sided"]==false);
    CHECK(inspected["materials"][0]["base_color"]["image"]==0);CHECK(inspected["models"][1]["selector"]["node"]==0);CHECK(inspected["models"][1]["triangles"]==4);
    CHECK(inspected["models"][1]["bounds"]["min"]==Json::array({5,0,0}));CHECK(inspected["models"][1]["bounds"]["max"]==Json::array({6,3,0}));
    gltf["buffers"][0]["uri"]="%2e%2e/quad.bin";write(root/"unsafe.gltf",gltf.dump());rejects([&]{foliage::inspect(root/"unsafe.gltf");});
    fs::remove_all(root);std::cout<<"Foliage download, archive, dependency and card-inspection contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';fs::remove_all(root);return 1;}}
