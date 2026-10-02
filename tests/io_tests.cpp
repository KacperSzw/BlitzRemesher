#include "blitz/io.hpp"
#include "../tools/neural/audit_settings.hpp"
#include "shared_vertices.hpp"
#include <fstream>
#include <iostream>
#include <bit>
using namespace blitz;
#define CHECK(x) do{if(!(x))throw std::runtime_error("line "+std::to_string(__LINE__)+": " #x);}while(0)
template<class F> void throws(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}CHECK(failed);}
static void color_imports(const std::filesystem::path& dir) {
    // The source values below are independent of the production quantizer.
    auto ply=[&](const char* type,const char* color) {
        std::ofstream f(dir/"colors.ply");
        f<<"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty "<<type<<" red\nproperty "<<type<<" green\nproperty "<<type<<" blue\nelement face 1\nproperty list uchar uint vertex_indices\nend_header\n";
        for(auto position:{"0 0 0 ","1 0 0 ","0 1 0 "})f<<position<<color<<'\n';f<<"3 0 1 2\n";
    };
    ply("ushort","0 32768 65535");CHECK((load_mesh(dir/"colors.ply").colors[0]==ColorRGBA8{0,128,255,255}));
    ply("float","0 0.5 1");CHECK((load_mesh(dir/"colors.ply").colors[0]==ColorRGBA8{0,128,255,255}));
    for(auto bad:{"-1 0 0","256 0 0","12.5 0 0"}){ply("uchar",bad);throws([&]{load_mesh(dir/"colors.ply");});}
    for(auto bad:{"-0.001 0 0","1.001 0 0","nan 0 0","inf 0 0"}){ply("float",bad);throws([&]{load_mesh(dir/"colors.ply");});}
    std::ofstream(dir/"colors.obj")<<"v 0 0 0 0 0.5 1\nv 1 0 0 0 0.5 1\nv 0 1 0 0 0.5 1\nf 1 2 3\n";
    CHECK((load_mesh(dir/"colors.obj").colors[0]==ColorRGBA8{0,128,255,255}));
    // glTF RGB/RGBA, normalized 16-bit and float accessors, including absent alpha.
    for(bool floats:{false,true})for(unsigned components:{3u,4u}) {
        std::vector<uint8_t> bytes;
        auto append=[&]<class T>(T v){auto b=std::bit_cast<std::array<uint8_t,sizeof(T)>>(v);if constexpr(std::endian::native==std::endian::big)std::reverse(b.begin(),b.end());bytes.insert(bytes.end(),b.begin(),b.end());};
        for(float v:{0.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f})append(v);
        for(uint32_t v:{0u,1u,2u})append(v);
        for(unsigned i=0;i<3;++i)for(unsigned c=0;c<components;++c) {
            if(floats)append(std::array<float,4>{0,.5f,1,.25f}[c]);
            else append(std::array<uint16_t,4>{0,32768,65535,16384}[c]);
        }
        // Six-byte RGB16 elements require an eight-byte glTF vertex stride.
        if(!floats&&components==3) {
            bytes.resize(48);
            for(unsigned i=0;i<3;++i){append(uint16_t(0));append(uint16_t(32768));append(uint16_t(65535));if(i!=2)append(uint16_t(0));}
        }
        nlohmann::json j={{"asset",{{"version","2.0"}}},{"buffers",nlohmann::json::array({{{"uri","colors.bin"},{"byteLength",bytes.size()}}})},
          {"bufferViews",nlohmann::json::array({{{"buffer",0},{"byteOffset",0},{"byteLength",36}},{{"buffer",0},{"byteOffset",36},{"byteLength",12}},{{"buffer",0},{"byteOffset",48},{"byteLength",bytes.size()-48}}})},
          {"accessors",nlohmann::json::array({{{"bufferView",0},{"componentType",5126},{"count",3},{"type","VEC3"}},{{"bufferView",1},{"componentType",5125},{"count",3},{"type","SCALAR"}},{{"bufferView",2},{"componentType",floats?5126:5123},{"count",3},{"type",components==3?"VEC3":"VEC4"}}})},
          {"meshes",nlohmann::json::array({{{"primitives",nlohmann::json::array({{{"attributes",{{"POSITION",0},{"COLOR_0",2}}},{"indices",1}}})}}})}};
        if(!floats)j["accessors"][2]["normalized"]=true;
        if(!floats&&components==3)j["bufferViews"][2]["byteStride"]=8;
        auto save=[&]{std::ofstream(dir/"colors.gltf")<<j.dump();std::ofstream f(dir/"colors.bin",std::ios::binary);f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());};
        save();auto imported=load_mesh(dir/"colors.gltf");
        for(auto c:imported.colors)CHECK((c==ColorRGBA8{0,128,255,uint8_t(components==4?64:255)}));
        if(floats)for(float invalid:{-.01f,1.01f,INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
            auto raw=std::bit_cast<uint32_t>(invalid);for(unsigned k=0;k<4;++k)bytes[48+k]=uint8_t(raw>>(k*8));
            save();throws([&]{load_mesh(dir/"colors.gltf");});
        }
    }
}
int main() {
    try {
        auto dir=std::filesystem::current_path()/"io-test-output";std::filesystem::create_directories(dir);
        color_imports(dir);
        Mesh m;m.positions={{0,0,0},{1,0,0},{0,1,0}};m.normals={{0,0,1},{0,0,1},{0,0,1}};m.uv={{0,0},{1,0},{0,1}};m.indices={0,1,2};
        save_ply(m.view(),dir/"triangle.ply");auto p=load_mesh(dir/"triangle.ply");CHECK(p.indices==m.indices);CHECK(p.uv[1].x==1);CHECK(p.normals[0].z==1);
        Result r;r.source=m.view();r.reference_bounds=bounds(m.view());r.max_changed_area=.5;
        for(int i=0;i<3;++i){Lod l;l.data.indices=m.indices;l.schedule={80./(i+1),.75+i*.25,1.+i};r.lods.push_back(std::move(l));}
        r.audit.profile=Profile::Attributes;r.audit.audit_views={7,2,135};r.audit.search_views={3,1,244};
        r.audit.weights={2,3,4};r.audit.normal_importance={{{0,.5},{1,.75}}};r.audit.attribute_importance={{{0,.25},{1,1}}};
        r.audit.search_supersample=2;r.audit.audit_supersample=4;r.audit.max_supersample=8;
        r.adaptive_retry_attempted=true;r.adaptive_retry_selected=true;r.adaptive_retry_evaluations=3;
        ProposalTrace retry_trace;retry_trace.pass=1;r.proposals.push_back(retry_trace);
        auto diagnostic=result_json(r);
        // Packed generation may retain only immutable LOD0 on cancellation or
        // an infeasible storage cap. Such results have no LOD chain to select.
        for(auto status:{Status::Cancelled,Status::BudgetLimited}) {
            auto incomplete=r;incomplete.status=status;incomplete.lods.resize(1);
            incomplete.candidates={{{uint32_t(m.view().triangles())},storage_stats(incomplete)}};
            auto summary=result_json(incomplete);
            CHECK(summary["status"]==(status==Status::Cancelled?"cancelled":"budget_limited"));
            CHECK(summary["lods"].size()==1&&summary["runtime_lod_count"]==1);
            CHECK(summary["candidates"][0]["triangles"].size()==1);
            CHECK(summary["selection_sweep"].empty());
        }
        CHECK(diagnostic["proposal_diagnostics"]["adaptive_retry_attempted"]==true);
        CHECK(diagnostic["proposal_diagnostics"]["adaptive_retry_selected"]==true);
        CHECK(diagnostic["proposal_diagnostics"]["adaptive_retry_evaluations"]==3);
        CHECK(diagnostic["proposals"][0]["pass"]=="adaptive_retry");
        {
            auto traced=r;traced.proposals.clear();
            for(uint8_t strategy:{uint8_t(4),uint8_t(5)}) {
                ProposalTrace proposal;proposal.strategy=strategy;proposal.origin=2;
                proposal.pass=3;proposal.gate=strategy==4?9:10;traced.proposals.push_back(proposal);
            }
            auto combined=result_json(traced);CHECK(combined["version"]==3);
            CHECK(combined["proposals"][0]["strategy"]=="neural_reuse");
            CHECK(combined["proposals"][1]["strategy"]=="neural_compact");
            CHECK(combined["proposals"][0]["origin"]=="tail_probe");
            CHECK(combined["proposals"][1]["pass"]=="graph_2");
            CHECK(combined["proposals"][0]["gate"]=="vertex_budget");
            CHECK(combined["proposals"][1]["gate"]=="objective_bound");
        }
        save_chain(r,dir/"gltf");auto g=load_mesh(dir/"gltf/chain.gltf");CHECK(g.indices.size()==3);CHECK(g.positions.size()==3);
        nlohmann::json manifest;std::ifstream(dir/"gltf/lods.json")>>manifest;CHECK(manifest["lods"][2]["gltf_mesh"]==0);
        CHECK(manifest["lods"].size()==3&&manifest["runtime_levels"]==nlohmann::json::array({0}));
        CHECK(manifest["runtime_storage"].size()==1&&manifest["runtime_storage"][0]["index_bytes"]==12);
        CHECK(manifest["max_changed_area"]==.5);
        CHECK(manifest["runtime_meshes"].size()==1&&manifest["runtime_meshes"][0]["scheduled_index"]==0);
        CHECK(manifest["audit_contract"]["scope"]=="configured_cameras"&&manifest["audit_contract"]["texture_images_scored"]==false);
        CHECK(manifest["audit_contract"]["profile"]=="attributes"&&manifest["audit_contract"]["audit_views"]["seed"]==135);
        CHECK(manifest["audit_contract"]["weights"]["normal"]==2&&manifest["audit_contract"]["normal_importance"][1][1]==.75);
        for(size_t i=0;i<3;++i)CHECK(manifest["lods"][i]["screen_pixels"]==80./(i+1)&&manifest["lods"][i]["transition_limit"]==.75+i*.25);
        nlohmann::json j;std::ifstream(dir/"gltf/chain.gltf")>>j;
        CHECK(j["meshes"].size()==1&&j["nodes"].size()==3&&j["nodes"][2]["mesh"]==0);
        // Collection-only foliage inspection must not relax the production
        // importer's opaque-material contract.
        for(auto mode:{"MASK","BLEND"}){auto alpha=j;alpha["materials"]=nlohmann::json::array({{{"alphaMode",mode},{"alphaCutoff",.41}}});alpha["meshes"][0]["primitives"][0]["material"]=0;
            std::ofstream(dir/"gltf/alpha.gltf")<<alpha.dump();throws([&]{load_gltf_mesh(dir/"gltf/alpha.gltf",0);});}
        j["nodes"][0]["scale"]={-1,2,3};std::ofstream(dir/"gltf/chain.gltf")<<j.dump();
        g=load_mesh(dir/"gltf/chain.gltf");CHECK(g.positions[1].x==-1);CHECK(g.positions[2].y==2);CHECK(g.indices[1]==2);
        auto n=cross(g.positions[g.indices[1]]-g.positions[0],g.positions[g.indices[2]]-g.positions[0]);CHECK(dot(n,g.normals[0])>0);
        g=load_gltf_mesh(dir/"gltf/chain.gltf",0);CHECK(g.positions[1].x==1&&g.positions[2].y==1&&g.indices==m.indices);
        bool bad_index=false;try{load_gltf_mesh(dir/"gltf/chain.gltf",1);}catch(...){bad_index=true;}CHECK(bad_index);
        // Equal triangle counts do not imply equal render data. A winding change
        // keeps the borrowed vertex accessors but requires a separate mesh.
        r.lods[2].data.indices={0,2,1};save_chain(r,dir/"distinct");
        std::ifstream(dir/"distinct/lods.json")>>manifest;
        CHECK(manifest["runtime_meshes"].size()==2&&manifest["runtime_meshes"][1]["scheduled_index"]==2);
        CHECK(manifest["runtime_meshes"][1]["screen_pixels"]==80./3&&manifest["runtime_meshes"][1]["shared_vertices"]==true);
        std::ifstream(dir/"distinct/chain.gltf")>>j;CHECK(j["meshes"].size()==2&&j["nodes"][2]["mesh"]==1);
        CHECK(j["meshes"][0]["primitives"][0]["attributes"]==j["meshes"][1]["primitives"][0]["attributes"]);
        g=load_gltf_mesh(dir/"distinct/chain.gltf",1);CHECK(g.indices==r.lods[2].data.indices);
        std::ofstream(dir/"triangle.obj")<<"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        auto o=load_mesh(dir/"triangle.obj");CHECK(o.indices.size()==3&&o.uv.size()==3);
        std::ofstream(dir/"triangle.stl")<<"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        CHECK(load_mesh(dir/"triangle.stl").indices.size()==3);
        auto config=settings_json(Settings{});CHECK(settings_json(settings_json(config))==config);
        for(uint32_t memory:{256u,8192u}) {
            NeuralOptions options;options.memory_mib=memory;
            auto visual=nlohmann::json{{"profile","coverage"},{"max_lod0_delta_px",2.5}};
            CHECK(model_audit_settings(visual,options).max_lod0_delta_px==2.5&&options.memory_mib==memory);
            auto input=visual;input["gpu_memory_mib"]=memory*2;
            auto parsed=model_audit_settings(input,options);
            CHECK(options.memory_mib==memory*2&&settings_json(parsed)==settings_json(settings_json(visual)));
            CHECK(input.at("gpu_memory_mib")==memory*2);
            input["typo_visual_setting"]=1;throws([&]{model_audit_settings(input,options);});
            for(auto bad:{nlohmann::json(0),nlohmann::json(-1),nlohmann::json(2.5),nlohmann::json("512"),nlohmann::json(uint64_t(UINT32_MAX)+1)}) {
                visual["gpu_memory_mib"]=bad;throws([&]{model_audit_settings(visual,options);});
            }
        }
        for(uint32_t trials:{1u,37u,65536u})for(uint8_t batch:{uint8_t(1),uint8_t(19),uint8_t(64)}) {
            NeuralOptions options;auto visual=nlohmann::json{{"profile","coverage"},{"max_changed_area",.25}};
            auto input=visual;input["action_trials"]=trials;input["action_batch"]=batch;
            CHECK(settings_json(model_audit_settings(input,options))==settings_json(settings_json(visual)));
            CHECK(options.action_trials==trials&&options.action_batch==batch);
            CHECK(input.contains("action_trials")&&input.contains("action_batch"));
        }
        for(const auto& key:{"action_trials","action_batch"})for(auto bad:{nlohmann::json(0),nlohmann::json(-1),nlohmann::json(1.5),nlohmann::json("8"),nlohmann::json(65537)}) {
            NeuralOptions options;throws([&]{model_audit_settings(nlohmann::json{{key,bad}},options);});
        }
        {NeuralOptions options;throws([&]{model_audit_settings(nlohmann::json{{"action_batch",65}},options);});}
        for(auto origin:{NeuralOrigin::Source,NeuralOrigin::Previous,NeuralOrigin::Both})for(bool preserve:{false,true}) {
            NeuralOptions options;auto visual=model_audit_settings(nlohmann::json{{"neural_origin",origin_name(origin)},{"preserve_uv",preserve}},options);
            CHECK(options.origin==origin&&options.preserve_uv==preserve&&settings_json(visual)==settings_json(Settings{}));
        }
        {NeuralOptions options;throws([&]{model_audit_settings(nlohmann::json{{"neural_origin","typo"}},options);});
            throws([&]{model_audit_settings(nlohmann::json{{"preserve_uv",1}},options);});}
        for(int stage:{0,1,2,3})for(bool screen:{false,true}) {
            auto parsed=settings_json(nlohmann::json{{"research",{{"appearance_stage",stage},{"conservative_screen",screen}}}});
            CHECK(unsigned(parsed.research.appearance_stage)==unsigned(stage)&&parsed.research.conservative_screen==screen);
            CHECK(settings_json(parsed)["research"]["appearance_stage"]==stage);
        }
        for(auto bad:nlohmann::json::array({-1,4,1.5,"1",true,nullptr}))
            throws([&]{settings_json(nlohmann::json{{"research",{{"appearance_stage",bad}}}});});
        for(auto bad:nlohmann::json::array({1,"true",nullptr}))
            throws([&]{settings_json(nlohmann::json{{"research",{{"conservative_screen",bad}}}});});
        for(auto key:{"density_targets","merge_wedges","shared_rebuild"}) {
            for(bool enabled:{false,true}) {
                auto controls=settings_json(nlohmann::json{{"research",{{key,enabled}}}});
                CHECK((std::string(key)=="density_targets"?controls.research.density_targets:std::string(key)=="merge_wedges"?controls.research.merge_wedges:controls.research.shared_rebuild)==enabled);
                CHECK(settings_json(controls)["research"][key]==enabled);
            }
            for(auto bad:nlohmann::json::array({1,"true",nullptr}))
                throws([&]{settings_json(nlohmann::json{{"research",{{key,bad}}}});});
        }
        for(int passes:{0,1,3}) {
            auto graph=settings_json(nlohmann::json{{"research",{{"graph_passes",passes}}}});
            CHECK(graph.research.graph_passes==passes&&settings_json(graph)["research"]["graph_passes"]==passes);
        }
        for(auto bad:nlohmann::json::array({-1,4,1.5,"1",true,nullptr}))
            throws([&]{settings_json(nlohmann::json{{"research",{{"graph_passes",bad}}}});});
        CHECK(config["max_added_vertex_bytes_bps"]==2000&&config["triangle_overhead_bps"]==0);
        for(auto cap:nlohmann::json::array({0,1000,2000,10000,nullptr})) {
            auto decoded=settings_json(nlohmann::json{{"max_added_vertex_bytes_bps",cap}});
            CHECK(settings_json(decoded)["max_added_vertex_bytes_bps"]==cap);
        }
        for(auto cap:nlohmann::json::array({-1,1.5,"2000",true,4294967295ull}))
            throws([&]{settings_json(nlohmann::json{{"max_added_vertex_bytes_bps",cap}});});
        CHECK(!config.contains("output")&&!config.contains("chain"));
        throws([&]{settings_json(nlohmann::json{{"output","reuse"}});});
        auto legacy=settings_json(nlohmann::json{{"output","reuse"},{"chain","progressive"}},true);
        CHECK(legacy.research.output==OutputMode::Reuse&&legacy.research.chain==ChainMode::Progressive);
        for(int b:{0,137,10000})CHECK(settings_json(nlohmann::json{{"triangle_overhead_bps",b}}).triangle_overhead_bps==b);
        throws([&]{settings_json(nlohmann::json{{"triangle_overhead_bps",2.5}});});
        for(double cap:{0.,.5,1.})CHECK(settings_json(nlohmann::json{{"max_changed_area",cap}}).max_changed_area==cap);
        for(double cap:{-.01,1.01})throws([&]{settings_json(nlohmann::json{{"max_changed_area",cap}});});
        auto fallback=settings_json(nlohmann::json{{"research",{{"topology_fallback",true}}}});
        for(int mib:{0,3,256}) {
            auto cached=settings_json(nlohmann::json{{"research",{{"coverage_cache_mib",mib}}}});
            CHECK(cached.research.coverage_cache_mib==mib&&settings_json(cached)["research"]["coverage_cache_mib"]==mib);
        }
        for(auto bad:nlohmann::json::array({-1,257,1.5,"64",true,nullptr}))
            throws([&]{settings_json(nlohmann::json{{"research",{{"coverage_cache_mib",bad}}}});});
        CHECK(fallback.research.topology_fallback&&settings_json(fallback)["research"]["topology_fallback"]==true);
        throws([&]{settings_json(nlohmann::json{{"objective","topology_relaxed"},{"research",{{"topology_fallback",true}}}});});
        CHECK(std::filesystem::file_size(dir/"distinct/chain.bin")==storage_stats(r).total());
        for(double weight:{0.,7.5,100.}){
            auto experiment=settings_json(nlohmann::json{{"profile","coverage"},{"research",{{"boundary_weight",weight},{"boundary_placement",true},{"adaptive_targets",true},{"independent_seams",true},{"trace",true}}}});
            CHECK(experiment.research.boundary_weight==weight&&experiment.research.boundary_placement&&experiment.research.adaptive_targets&&experiment.research.independent_seams&&experiment.research.trace);
            auto encoded=settings_json(experiment);CHECK(settings_json(settings_json(encoded))==encoded);
        }
        throws([&]{settings_json(nlohmann::json{{"research",{{"unknown_option",true}}}});});
        bool failed=false;try{settings_json({{"levels",258}});}catch(...){failed=true;}CHECK(failed);
        Mesh colors;
        for(unsigned k=0;k<256;++k){colors.positions.push_back({float(k%16),float(k/16),0});colors.colors.push_back({uint8_t(k),uint8_t(255-k),37,uint8_t(k)});}
        colors.indices={0,1,16};save_ply(colors.view(),dir/"bytes.ply");
        CHECK(load_mesh(dir/"bytes.ply").colors==colors.colors);
        Result cr;cr.source=colors.view();cr.lods.resize(2);for(auto& l:cr.lods)l.data.indices=colors.indices;
        save_chain(cr,dir/"byte-gltf");CHECK(load_mesh(dir/"byte-gltf/chain.gltf").colors==colors.colors);
        std::ifstream(dir/"byte-gltf/chain.gltf")>>j;auto color_index=j["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"].get<size_t>();
        auto& color_accessor=j["accessors"][color_index];CHECK(color_accessor["componentType"]==5121&&color_accessor["normalized"]==true);
        auto& color_view=j["bufferViews"][color_accessor["bufferView"].get<size_t>()];
        CHECK(color_view["byteLength"]==colors.colors.size()*4&&color_view["byteOffset"].get<size_t>()%4==0);
        {
            Mesh input;input.positions={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0},{0,0,0},{.2f,0,0}};
            input.indices={0,1,4,1,2,5,2,3,5,3,0,4,4,1,5,4,5,3};input.double_sided={1};
            for(auto p:input.positions){input.normals.push_back({0,0,1});input.uv.push_back({(p.x+1)*.5f,(p.y+1)*.5f});input.colors.push_back({40,90,120,77});input.tangents.push_back({1,0,0,-1});}
            detail::SourceVertices table(input.view(),true);Result mixed;mixed.source=input.view();mixed.reference_bounds=bounds(input.view());
            Lod base;base.data.indices=input.indices;mixed.lods.push_back(base);
            for(float x:{.1f,.3f}){Lod l;l.shared_vertices=false;l.data=input;l.data.indices={0,1,4,1,2,4,2,3,4,3,0,4};
                l.data.positions[4]={x,0,0};l.data.uv[4]={(x+1)*.5f,.5f};compact(l.data);table.share(l);mixed.lods.push_back(std::move(l));}
            detail::share_result_vertices(mixed);save_chain(mixed,dir/"mixed");
            const uint64_t expected_bytes=8*(12+12+8+4+16)+(18+12+12)*4;
            CHECK(std::filesystem::file_size(dir/"mixed/chain.bin")==expected_bytes&&storage_stats(mixed).total()==expected_bytes);
            nlohmann::json gltf;std::ifstream(dir/"mixed/chain.gltf")>>gltf;
            auto first=gltf["meshes"][0]["primitives"][0]["attributes"],second=gltf["meshes"][1]["primitives"][0]["attributes"],third=gltf["meshes"][2]["primitives"][0]["attributes"];
            CHECK(second==third);
            for(auto it=first.begin();it!=first.end();++it){auto a=gltf["accessors"][it.value().get<size_t>()],b=gltf["accessors"][second[it.key()].get<size_t>()];
                CHECK(a["count"]==6&&b["count"]==8&&a["bufferView"]==b["bufferView"]);}
            CHECK(load_mesh(dir/"mixed/chain.gltf").positions.size()==6);
        }
        std::cout<<"Import, transforms, runtime mesh sharing, shared accessors and settings round trips passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
