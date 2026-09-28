#pragma once
#include "remesher.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
namespace blitz {
Mesh load_mesh(const std::filesystem::path&);
// Read one glTF mesh in mesh-local coordinates, without scene transforms.
Mesh load_gltf_mesh(const std::filesystem::path&,size_t mesh_index);
void save_ply(MeshView,const std::filesystem::path&);
void save_chain(const Result&,const std::filesystem::path&);
nlohmann::json result_json(const Result&);
Settings settings_json(const nlohmann::json&);
nlohmann::json settings_json(const Settings&);
}
