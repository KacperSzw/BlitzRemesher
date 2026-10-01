#pragma once
#include "training/checkpoint.hpp"
namespace blitz::neural::training {
// The checkpoint journal contains only the resident window and live progress.
// Completed phases remain append-only, so publication cost does not grow with
// the number of completed teacher/update cycles.
class CycleHistory {
    fs::path path_;
    std::ofstream stream_;
    uint64_t bytes_{};

  public:
    CycleHistory(fs::path path, uint64_t durable_bytes)
        : path_(std::move(path)), bytes_(durable_bytes) {
        uint64_t actual = fs::exists(path_) ? fs::file_size(path_) : 0;
        if (actual < durable_bytes)
            throw std::invalid_argument("cycle history shorter than recovery journal");
        if (actual > durable_bytes) {
            // Keep uncommitted work as evidence without replaying it twice.
            std::ifstream input(path_, std::ios::binary);
            input.seekg(durable_bytes);
            auto orphan = path_;
            orphan += ".orphan-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            std::ofstream output(orphan, std::ios::binary);
            output << input.rdbuf();
            output.close();
            if (!output)
                throw std::runtime_error("history orphan write failed");
            fs::resize_file(path_, durable_bytes);
        }
        stream_.open(path_, std::ios::binary | std::ios::app);
        if (!stream_)
            throw std::runtime_error("cycle history open failed");
    }
    void append(const char* kind, const json& value) {
        auto line = json{{"kind", kind}, {"value", value}}.dump();
        stream_ << line << '\n';
        if (!stream_)
            throw std::runtime_error("cycle history write failed");
        bytes_ += line.size() + 1;
    }
    uint64_t sync() {
        stream_.flush();
        if (!stream_)
            throw std::runtime_error("cycle history flush failed");
        sync_checkpoint_file(path_);
        return bytes_;
    }
    void assemble(json& report) {
        sync();
        for (auto key : {"phases", "failed_conditions", "empty_conditions",
                         "unavailable_conditions", "all_datasets"})
            report[key] = json::array();
        std::ifstream input(path_);
        std::string line;
        while (std::getline(input, line)) {
            auto record = json::parse(line);
            auto key = record.at("kind").get<std::string>();
            report.at(key).push_back(record.at("value"));
        }
        if (!input.eof())
            throw std::runtime_error("cycle history read failed");
    }
};
} // namespace blitz::neural::training
