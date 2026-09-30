#include "neural_packing.hpp"
#include "neural_action_gpu.hpp"
#include "neural_audit_io.hpp"
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
int main(int argc,char** argv){try{
    if((argc==6||argc==7)&&std::string_view(argv[1])=="--repair"){
        if(fs::exists(argv[4]))throw std::invalid_argument("choose a fresh report");auto mesh=load_mesh(argv[2]);auto s=settings_json(read_json(argv[3]));auto b=bounds(mesh.view());auto steps=schedule(b,s);NeuralOptions o;o.raster_backend=NeuralRasterBackend::Vulkan;o.memory_mib=512;o.view_batch=1;AuditSession session(o);
        EvalSettings e;e.profile=s.profile;e.weights=s.weights;e.screen_size=steps[1].pixels;e.limit=steps[1].source;e.max_changed_area=s.max_changed_area;e.views=s.audit_views;e.supersample=s.audit_supersample;e.max_supersample=s.max_supersample;
        auto search=e;search.views=s.search_views;search.supersample=s.search_supersample;search.max_changed_area=1;
        Mesh prepared;if(argc==7)prepared=audit_mesh(read_json(argv[6]).at("candidate"));
        auto fixed=repair_packing_gpu(mesh.view(),o,e,neural_unsigned(argv[5]),prepared.view(),&search);auto q=vertex_bounds(mesh.view());AuditCuda audit(o,mesh.view());GpuActionState state(fixed.mesh.view(),o,true,&q);auto snapshot=state.snapshot();
        json attempts=json::array();for(auto a:fixed.attempts)attempts.push_back({{"trial",a.trial},{"view",a.view},{"vertex",a.vertex},{"axis",a.axis},{"direction",a.direction},{"before",a.before},{"after",a.after},{"kept",a.kept},{"patch_face",a.patch_face==UINT32_MAX?json(nullptr):json(a.patch_face)}});
        auto directory=fs::path(argv[4]);fs::create_directories(directory);write_json(directory/"report.json",{{"initial",measurement_json(fixed.initial)},{"repaired",measurement_json(fixed.final)},{"device",measurement_json(audit.evaluate(mesh.view(),state.view(),b,e))},{"snapshot",measurement_json(audit.evaluate(mesh.view(),snapshot.data.view(),b,e))},{"snapshot_search",measurement_json(audit.evaluate(mesh.view(),snapshot.data.view(),b,search))},{"trials",fixed.trials},{"attempts",attempts}});
        NeuralStats stats;auto& f=stats.confirmation_failure.emplace();f.source=mesh;f.reference=mesh;f.candidate=fixed.mesh;f.bounds=b;f.settings=e;f.raster=o.raster_backend;f.storage=o.draw_storage();f.backend=NeuralConfirmation::Gpu;f.gpu=fixed.final;save_audit_failure(stats,directory/"replay.json");
        if(!fixed.final.passed){auto camera=cameras(b,e.screen_size,e.views).at(fixed.final.worst_view);auto full=o;full.vertex_storage=NeuralVertexStorage::Float32;auto x=diagnostic_raster(mesh.view(),b,camera,e.screen_size,e.supersample,false,full),y=diagnostic_raster(fixed.mesh.view(),b,camera,e.screen_size,e.supersample,false,o,&q);auto witnesses=packing_witnesses(x,y,e);write_json(directory/"witnesses.json",{{"pixels",witnesses.pixels},{"faces",witnesses.faces},{"unmatched",witnesses.unmatched}});}
        return 0;
    }
    if(argc<3||argc>4)throw std::invalid_argument("blitz-neural-packing-diagnostic ASSET FRESH_REPORT [PIXELS], or --repair MESH SETTINGS FRESH_DIRECTORY BUDGET [PREPARED_REPLAY]");if(fs::exists(argv[2]))throw std::invalid_argument("choose a fresh report");auto [mesh,metadata]=training_mesh(argv[1]);NeuralOptions o;o.raster_backend=NeuralRasterBackend::Vulkan;o.memory_mib=512;auto e=action_eval(argc==4?std::stod(argv[3]):128);json result={{"asset",metadata},{"complete",false},{"score",nullptr},{"controls",json::array()}};
    for(auto storage:{NeuralVertexStorage::Float32,NeuralVertexStorage::Position16,NeuralVertexStorage::Packed}){o.vertex_storage=storage;auto report=diagnose_packing(mesh.view(),mesh.view(),o,e);result["controls"].push_back(report);write_json(argv[2],result);}
    result["complete"]=true;write_json(argv[2],result);std::cout<<"packing diagnostics written to "<<argv[2]<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
