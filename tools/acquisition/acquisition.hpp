#pragma once
#include <cstdint>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace blitz::acquisition {
inline constexpr uint64_t max_file = 1024ull * 1024 * 1024;
std::string request(const std::string& url);
void write(const std::filesystem::path&, const std::string&);
std::string read(const std::filesystem::path&);
std::string hash(const std::string&);
nlohmann::json cached(const std::string& url, const std::filesystem::path&);
bool contains(const nlohmann::json&, const std::string&);
bool woody(const std::string&);
std::string safe(std::string);
nlohmann::json poly(const std::string& id, const std::string& category,
                    const std::filesystem::path& root);
std::string extract_obj(const std::string& bytes);
nlohmann::json smithsonian(const nlohmann::json&, const std::filesystem::path& root);
} // namespace blitz::acquisition
