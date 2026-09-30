#include "neural_packing.hpp"
#include <iostream>
using namespace blitz;using namespace blitz::neural;using namespace blitz::neural::training;
int main(int argc,char** argv){try{
    if(argc<3||argc>4)throw std::invalid_argument("blitz-neural-packing-diagnostic ASSET FRESH_REPORT [PIXELS]");if(fs::exists(argv[2]))throw std::invalid_argument("choose a fresh report");auto [mesh,metadata]=training_mesh(argv[1]);NeuralOptions o;o.raster_backend=NeuralRasterBackend::Vulkan;o.memory_mib=512;auto e=action_eval(argc==4?std::stod(argv[3]):128);json result={{"asset",metadata},{"complete",false},{"score",nullptr},{"controls",json::array()}};
    for(auto storage:{NeuralVertexStorage::Float32,NeuralVertexStorage::Position16,NeuralVertexStorage::Packed}){o.vertex_storage=storage;auto report=diagnose_packing(mesh.view(),mesh.view(),o,e);result["controls"].push_back(report);write_json(argv[2],result);}
    result["complete"]=true;write_json(argv[2],result);std::cout<<"packing diagnostics written to "<<argv[2]<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
