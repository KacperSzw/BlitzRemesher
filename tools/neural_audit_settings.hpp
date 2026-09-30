#pragma once
#include "blitz/io.hpp"
#include "blitz/neural.hpp"

// The workspace limit belongs to the audit runner, not the visual settings.
inline blitz::Settings model_audit_settings(nlohmann::json config,blitz::NeuralOptions& options) {
    if(auto it=config.find("gpu_memory_mib");it!=config.end()) {
        if(!it->is_number_integer()||*it<=0||*it>UINT32_MAX)throw std::invalid_argument("invalid audit workspace limit");
        options.memory_mib=it->get<uint32_t>();config.erase(it);
    }
    return blitz::settings_json(config);
}
