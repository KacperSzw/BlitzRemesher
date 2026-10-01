#pragma once
#include "blitz/io.hpp"
#include "blitz/neural.hpp"
#include "training/json.hpp"

// Execution budgets are deliberately separate from the visual contract.
inline blitz::Settings model_audit_settings(nlohmann::json config, blitz::NeuralOptions& options) {
    auto take = [&](const char* name, uint32_t low, uint32_t high, uint32_t current) {
        auto it = config.find(name);
        if (it == config.end())
            return current;
        if (!it->is_number_integer() || *it < low || *it > high)
            throw std::invalid_argument(std::string("invalid audit ") + name);
        auto value = it->get<uint32_t>();
        config.erase(it);
        return value;
    };
    options.memory_mib = take("gpu_memory_mib", 128, 65536, options.memory_mib);
    options.action_trials = take("action_trials", 1, 65536, options.action_trials);
    options.action_batch = uint8_t(take("action_batch", 1, 64, options.action_batch));
    if (auto it = config.find("neural_origin"); it != config.end()) {
        options.origin = origin_option(it->get<std::string>());
        config.erase(it);
    }
    if (auto it = config.find("preserve_uv"); it != config.end()) {
        if (!it->is_boolean())
            throw std::invalid_argument("audit preserve_uv must be boolean");
        options.preserve_uv = it->get<bool>();
        config.erase(it);
    }
    return blitz::settings_json(config);
}
