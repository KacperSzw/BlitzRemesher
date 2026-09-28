#pragma once
#include <nlohmann/json.hpp>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
namespace foliage {
using Json=nlohmann::json;
using Fetch=std::function<std::string(const std::string&)>;
std::string sha256(std::string_view);
std::filesystem::path relative_path(const std::string&);
Json acquire_file(const std::filesystem::path&,const Json&,const Fetch&);
Json extract_archive(const std::filesystem::path&,const std::filesystem::path&);
Json inspect(const std::filesystem::path&,bool allow_missing_mtl=false);
Json check(const Json&,const std::filesystem::path&);
int main(int,char**);
}
