#pragma once
#include "training/action_types.hpp"
#include "training/compact.hpp"
namespace blitz::neural::training {
inline constexpr uint32_t compact_data_version = 2;
struct CompactActions {
    uint32_t architecture{placement_schema};
    std::vector<uint16_t> values;
    std::vector<uint8_t> colors, labels;
    std::vector<uint32_t> flags, offsets{0}, target_offsets{0}, escape_ids, from, to;
    std::vector<float> conditions, progress, escape_values;
    std::vector<int16_t> targets;
    size_t states() const {
        return offsets.size() - 1;
    }
    size_t bytes() const {
        return values.size() * 2 + colors.size() + labels.size() + targets.size() * 2 +
               (flags.size() + offsets.size() + target_offsets.size() + escape_ids.size() +
                from.size() + to.size() + conditions.size() + progress.size() +
                escape_values.size()) *
                   4;
    }
    CompactView view(uint32_t state) const {
        return {values.data(),
                colors.data(),
                flags.data(),
                conditions.data() + size_t(state) * action_condition_width(architecture),
                escape_ids.data(),
                escape_values.data(),
                uint32_t(escape_ids.size()),
                offsets[state],
                target_offsets.data(),
                targets.data(),
                action_condition_width(architecture)};
    }
};
inline CompactActions compact_actions(const ActionData& data) {
    validate_actions(data);
    CompactActions out;
    out.architecture = data.architecture;
    out.offsets = data.offsets;
    out.progress = data.progress;
    out.labels = data.labels;
    out.from = data.from;
    out.to = data.to;
    auto width = policy_inputs(data.architecture);
    auto condition_width = action_condition_width(data.architecture);
    out.values.resize(data.labels.size() * compact_width(width));
    out.colors.resize(data.labels.size() * color_width(width));
    out.flags.resize(data.labels.size());
    out.conditions.resize(data.states() * condition_width);
    for (uint32_t state = 0; state < data.states(); ++state) {
        bool condition_set = false;
        for (uint32_t row = data.offsets[state]; row < data.offsets[state + 1]; ++row) {
            auto* x = data.x.data() + size_t(row) * width;
            for (uint32_t c = 0; c < width; ++c) {
                int slot = compact_slot(c);
                float value = x[c];
                if (c == 79 && condition_width == 9) {
                    auto& shared = out.conditions[size_t(state) * condition_width + 8];
                    if (condition_set && shared != value)
                        throw std::invalid_argument("preserve-UV conditions differ");
                    shared = value;
                    continue;
                }
                if (c == 78 || c == 79) {
                    if (value != x[c == 78 ? 23 : 47])
                        throw std::invalid_argument("duplicate feature differs");
                    continue;
                }
                if (slot >= -32 && slot < 0) {
                    if (value != 0 && value != 1)
                        throw std::invalid_argument("feature flag outside 0..1");
                    if (value == 1)
                        out.flags[row] |= 1u << unsigned(-slot - 1);
                    continue;
                }
                if (slot <= -33 && slot >= -40) {
                    auto& shared =
                        out.conditions[size_t(state) * condition_width + unsigned(-slot - 33)];
                    if (condition_set && shared != value)
                        throw std::invalid_argument("state conditions differ");
                    shared = value;
                    continue;
                }
                bool escape = false;
                auto encoding = feature_encoding(c);
                if (encoding == FeatureEncoding::Color) {
                    unsigned q =
                        value >= 0 && value <= 1 ? unsigned(std::lround(double(value) * 255)) : 255;
                    escape = value != float(q) / 255.f;
                    out.colors[size_t(row) * color_width(width) + unsigned(-slot - 41)] =
                        uint8_t(escape ? 255 : q);
                } else {
                    uint16_t q;
                    if (encoding == FeatureEncoding::Snorm) {
                        escape = value < -1 || value > 1;
                        q = escape ? 0x8000 : uint16_t(int16_t(std::lround(double(value) * 32767)));
                    } else if (encoding == FeatureEncoding::Unorm) {
                        escape = value < 0 || value > 1;
                        q = escape ? 65535 : uint16_t(std::lround(double(value) * 65535));
                    } else {
                        q = half_bits(std::bit_cast<uint32_t>(value));
                        escape = (q & 0x7c00) == 0x7c00 || (value != 0 && (q & 0x7fff) == 0);
                        if (escape)
                            q = 0x7e00;
                    }
                    out.values[size_t(row) * compact_width(width) + unsigned(slot)] = q;
                }
                if (escape) {
                    out.escape_ids.push_back(row * packed_width(width) + uint32_t(feature_slot(c)));
                    out.escape_values.push_back(value);
                }
            }
            condition_set = true;
            if (is_placement_schema(data.architecture))
                for (unsigned group = 0; group < 3; ++group)
                    if (data.labels[row] & (32u << group))
                        for (unsigned c = 0; c < 3; ++c)
                            out.targets.push_back(int16_t(
                                std::lround(double(data.targets[size_t(row) * 9 + group * 3 + c]) /
                                            (group ? 2 : 1) * 32767)));
            out.target_offsets.push_back(uint32_t(out.targets.size()));
        }
    }
    return out;
}
inline void validate_compact(const CompactActions& a) {
    auto width = policy_inputs(a.architecture);
    size_t rows = a.labels.size();
    if (!width || a.offsets.empty() || a.offsets[0] || a.offsets.back() != rows ||
        !std::is_sorted(a.offsets.begin(), a.offsets.end()) ||
        a.values.size() != rows * compact_width(width) ||
        a.colors.size() != rows * color_width(width) || a.flags.size() != rows ||
        a.conditions.size() != a.states() * action_condition_width(a.architecture) ||
        a.progress.size() != a.states() || a.from.size() != rows || a.to.size() != rows ||
        a.target_offsets.size() != rows + 1 || a.target_offsets[0] ||
        a.target_offsets.back() != a.targets.size() ||
        !std::is_sorted(a.target_offsets.begin(), a.target_offsets.end()) ||
        a.escape_ids.size() != a.escape_values.size())
        throw std::invalid_argument("compact shard dimensions");
    for (size_t i = 0; i < a.states(); ++i)
        if (a.offsets[i + 1] - a.offsets[i] > action_pool || !std::isfinite(a.progress[i]) ||
            a.progress[i] < 0 || a.progress[i] > 1)
            throw std::invalid_argument("compact state dimensions/progress");
    for (size_t i = 0; i < a.escape_ids.size(); ++i)
        if (a.escape_ids[i] >= rows * packed_width(width) ||
            (i && a.escape_ids[i] <= a.escape_ids[i - 1]) || !std::isfinite(a.escape_values[i]))
            throw std::invalid_argument("compact escape table");
    for (float c : a.conditions)
        if (!std::isfinite(c))
            throw std::invalid_argument("nonfinite compact condition");
    if (a.architecture == conditioned_placement_schema)
        for (size_t state = 0; state < a.states(); ++state)
            if (a.conditions[state * 9 + 8] != 0 && a.conditions[state * 9 + 8] != 1)
                throw std::invalid_argument("invalid compact preserve-UV condition");
    for (auto q : a.targets)
        if (q == INT16_MIN)
            throw std::invalid_argument("invalid compact target");
    for (size_t row = 0; row < rows; ++row) {
        auto label = a.labels[row];
        unsigned groups = 0;
        for (unsigned g = 0; g < 3; ++g)
            groups += bool(label & (32u << g));
        if (a.target_offsets[row + 1] - a.target_offsets[row] !=
            (is_placement_schema(a.architecture) ? groups * 3 : 0))
            throw std::invalid_argument("compact target mask differs");
        if (a.architecture == action_schema) {
            if (label > 15 || !(label & Queried) || ((label & Preferred) && ((label & 3) != 3)))
                throw std::invalid_argument("invalid compact action mask");
        } else if (((label & 1) && !(label & SourceKnown)) ||
                   ((label & 2) && !(label & AdjacentKnown)) ||
                   ((label & (Preferred | PositionKnown)) && ((label & 27) != 27)) ||
                   ((label & (Normal0Known | Normal1Known)) && !(label & PositionKnown)))
            throw std::invalid_argument("invalid compact placement mask");
    }
    // Validate every decoded input, including reserved half codes and missing
    // exceptions. Metadata is untrusted until this bounded CPU check completes.
    for (uint32_t s = 0; s < a.states(); ++s) {
        auto v = a.view(s);
        for (uint32_t r = 0; r < a.offsets[s + 1] - a.offsets[s]; ++r)
            for (uint32_t c = 0; c < width; ++c) {
                auto slot = compact_slot(c);
                if (slot >= 0 && feature_encoding(c) == FeatureEncoding::Snorm &&
                    a.values[size_t(v.first + r) * compact_width(width) + unsigned(slot)] ==
                        0x8000 &&
                    !std::binary_search(a.escape_ids.begin(), a.escape_ids.end(),
                                        (v.first + r) * packed_width(width) +
                                            uint32_t(feature_slot(c))))
                    throw std::invalid_argument("missing normalized feature exception");
                if (!std::isfinite(compact_feature(v, r, c, width)))
                    throw std::invalid_argument("nonfinite compact feature");
            }
    }
}
inline CompactActions supervised_compact(const CompactActions& a) {
    CompactActions out;
    out.architecture = a.architecture;
    out.conditions = a.conditions;
    out.progress = a.progress;
    auto width = policy_inputs(a.architecture), cw = compact_width(width), rgb = color_width(width),
         pw = packed_width(width);
    for (uint32_t s = 0; s < a.states(); ++s) {
        for (uint32_t r = a.offsets[s]; r < a.offsets[s + 1]; ++r) {
            if (!(a.labels[r] & (is_placement_schema(a.architecture) ? 248 : 8)))
                continue;
            auto row = uint32_t(out.labels.size());
            out.values.insert(out.values.end(), a.values.begin() + size_t(r) * cw,
                              a.values.begin() + size_t(r + 1) * cw);
            out.colors.insert(out.colors.end(), a.colors.begin() + size_t(r) * rgb,
                              a.colors.begin() + size_t(r + 1) * rgb);
            out.flags.push_back(a.flags[r]);
            out.labels.push_back(a.labels[r]);
            out.from.push_back(a.from[r]);
            out.to.push_back(a.to[r]);
            out.targets.insert(out.targets.end(), a.targets.begin() + a.target_offsets[r],
                               a.targets.begin() + a.target_offsets[r + 1]);
            out.target_offsets.push_back(uint32_t(out.targets.size()));
            auto begin = std::lower_bound(a.escape_ids.begin(), a.escape_ids.end(), r * pw),
                 end = std::lower_bound(begin, a.escape_ids.end(), (r + 1) * pw);
            for (auto it = begin; it != end; ++it) {
                out.escape_ids.push_back(row * pw + *it - r * pw);
                out.escape_values.push_back(a.escape_values[size_t(it - a.escape_ids.begin())]);
            }
        }
        out.offsets.push_back(uint32_t(out.labels.size()));
    }
    return out;
}
inline ActionData expand_compact(const CompactActions& a) {
    validate_compact(a);
    ActionData out;
    out.architecture = a.architecture;
    out.labels = a.labels;
    out.offsets = a.offsets;
    out.progress = a.progress;
    out.from = a.from;
    out.to = a.to;
    auto width = policy_inputs(a.architecture);
    out.x.reserve(a.labels.size() * width);
    if (is_placement_schema(a.architecture))
        out.targets.reserve(a.labels.size() * 9);
    for (uint32_t s = 0; s < a.states(); ++s) {
        auto v = a.view(s);
        for (uint32_t row = 0; row < a.offsets[s + 1] - a.offsets[s]; ++row) {
            for (uint32_t c = 0; c < width; ++c)
                out.x.push_back(compact_feature(v, row, c, width));
            if (is_placement_schema(a.architecture))
                for (unsigned c = 0; c < 9; ++c)
                    out.targets.push_back(compact_target(v, row, a.labels[a.offsets[s] + row], c));
        }
    }
    validate_actions(out);
    return out;
}
inline void write_compact(std::ostream& f, const CompactActions& a) {
    validate_compact(a);
    f.write(a.architecture == conditioned_placement_schema ? "BLZACT05" : "BLZACT04", 8);
    f.write(reinterpret_cast<const char*>(&a.architecture), 4);
    write_vector(f, a.values);
    write_vector(f, a.colors);
    write_vector(f, a.flags);
    write_vector(f, a.labels);
    write_vector(f, a.offsets);
    write_vector(f, a.conditions);
    write_vector(f, a.progress);
    write_vector(f, a.target_offsets);
    write_vector(f, a.targets);
    write_vector(f, a.escape_ids);
    write_vector(f, a.escape_values);
    write_vector(f, a.from);
    write_vector(f, a.to);
}
inline CompactActions read_compact(std::istream& f, bool conditioned = false) {
    CompactActions a;
    f.read(reinterpret_cast<char*>(&a.architecture), 4);
    if (!policy_inputs(a.architecture) ||
        (a.architecture == conditioned_placement_schema) != conditioned)
        throw std::invalid_argument("compact architecture");
    constexpr uint64_t rows = action_pool * 65536ull;
    a.values = read_vector<uint16_t>(f, rows * 74);
    a.colors = read_vector<uint8_t>(f, rows * 12);
    a.flags = read_vector<uint32_t>(f, rows);
    a.labels = read_vector<uint8_t>(f, rows);
    a.offsets = read_vector<uint32_t>(f, 65537);
    a.conditions = read_vector<float>(f, 65536 * action_condition_width(a.architecture));
    a.progress = read_vector<float>(f, 65536);
    a.target_offsets = read_vector<uint32_t>(f, rows + 1);
    a.targets = read_vector<int16_t>(f, rows * 9);
    a.escape_ids = read_vector<uint32_t>(f, rows * 86);
    a.escape_values = read_vector<float>(f, rows * 86);
    a.from = read_vector<uint32_t>(f, rows);
    a.to = read_vector<uint32_t>(f, rows);
    if (f.peek() != EOF)
        throw std::invalid_argument("unexpected compact shard tail");
    validate_compact(a);
    return a;
}
} // namespace blitz::neural::training
