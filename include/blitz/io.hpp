#pragma once
#include "remesher.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
namespace blitz {
Mesh load_mesh(const std::filesystem::path&);
void save_ply(MeshView,const std::filesystem::path&);
void save_chain(const Result&,const std::filesystem::path&);
nlohmann::json result_json(const Result&);
Settings settings_json(const nlohmann::json&);
nlohmann::json settings_json(const Settings&);
}
