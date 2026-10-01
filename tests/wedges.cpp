#include "wedges.hpp"
#include <iostream>
#include <stdexcept>
using namespace blitz;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +      \
                                     ": " #x);                                                     \
    } while (0)
static Mesh fan() {
    Mesh m;
    m.positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 0}, {.2f, 0, 0}};
    m.indices = {0, 1, 4, 1, 2, 5, 2, 3, 5, 3, 0, 4, 4, 1, 5, 4, 5, 3};
    m.double_sided = {1};
    for (auto p : m.positions) {
        m.normals.push_back({0, 0, 1});
        m.uv.push_back({(p.x + 1) * .5f, (p.y + 1) * .5f});
        m.colors.push_back({71, 133, 211, 77});
        m.tangents.push_back({1, 0, 0, -1});
    }
    return m;
}
int main() {
    try {
        const auto source = fan(), original = source;
        auto emitted = source;
        emitted.positions[5] = emitted.positions[4];
        emitted.indices.resize(12);
        const uint32_t positions[] = {0, 1, 2, 3, 4, 5}, clusters[] = {0, 1, 2, 3, 4, 4},
                       faces[] = {0, 1, 2, 3};
        ReductionStats stats;
        CHECK(detail::merge_wedges(source.view(), positions, clusters, faces, emitted, &stats));
        CHECK(validate(emitted.view()).empty() && emitted.positions.size() == 5 &&
              emitted.indices.size() == 12);
        CHECK(same_mesh_data(source.view(), original.view()) && stats.appearance_bytes > 0);
        for (size_t i = 0; i < emitted.positions.size(); ++i) {
            auto p = emitted.positions[i], n = emitted.normals[i];
            auto t = emitted.tangents[i];
            CHECK(std::abs(emitted.uv[i].x - (p.x + 1) * .5) < 1e-6 &&
                  std::abs(emitted.uv[i].y - (p.y + 1) * .5) < 1e-6);
            CHECK(emitted.colors[i] == ColorRGBA8({71, 133, 211, 77}));
            CHECK(std::abs(length(n) - 1) < 1e-6 && std::abs(dot(n, {t.x, t.y, t.z})) < 1e-6 &&
                  t.w == -1);
        }
        CHECK(uv_distortion(emitted.view()).negative_uv_faces == 0);
        // Precision survives fitting even when only the nonrepresentative member
        // of the merged cluster requires exact position storage.
        for (uint32_t marked : {0u, 4u, 5u}) {
            auto precise = source;
            precise.exact_position_bits = {1u << marked};
            const auto original_precise = precise;
            auto fitted = precise;
            fitted.positions[5] = fitted.positions[4];
            fitted.indices.resize(12);
            CHECK(
                detail::merge_wedges(precise.view(), positions, clusters, faces, fitted, nullptr));
            CHECK(validate(fitted.view()).empty() &&
                  same_mesh_data(precise.view(), original_precise.view()));
            for (size_t i = 0; i < fitted.positions.size(); ++i) {
                auto expected_position = precise.positions[marked == 5 ? 4 : marked];
                CHECK(fitted.view().exact_position(i) ==
                      (length(fitted.positions[i] - expected_position) < 1e-6));
            }
        }
        // Discrete alpha/handedness differences retain separate wedges.
        for (bool alpha : {false, true}) {
            auto discrete = source;
            if (alpha)
                discrete.colors[5].a = 99;
            else
                discrete.tangents[5].w = 1;
            auto out = discrete;
            out.positions[5] = out.positions[4];
            out.indices.resize(12);
            CHECK(detail::merge_wedges(discrete.view(), positions, clusters, faces, out, nullptr));
            CHECK(out.positions.size() == 6);
            bool different = false;
            for (size_t i = 0; i < out.positions.size(); ++i)
                different |= alpha ? out.colors[i].a == 99 : out.tangents[i].w == 1;
            CHECK(different);
        }
        // Two sheets touch only at one original index. Corner fans must split that
        // point instead of introducing a bridge through a vertex-only contact.
        Mesh touch;
        touch.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
        touch.indices = {0, 1, 2, 0, 3, 4};
        touch.exact_position_bits = {1};
        auto out = touch;
        const uint32_t ids[] = {0, 1, 2, 3, 4}, both[] = {0, 1};
        CHECK(detail::merge_wedges(touch.view(), ids, ids, both, out, nullptr));
        CHECK(out.positions.size() == 6 && out.indices[0] != out.indices[3]);
        CHECK(out.view().exact_position(out.indices[0]) &&
              out.view().exact_position(out.indices[3]) && validate(out.view()).empty());
        // A duplicated interior endpoint belongs to a separate chart. Its blue
        // wedge must survive beside the red one at the same collapsed position.
        auto seam = source;
        seam.colors.assign(6, {255, 0, 0, 77});
        for (auto i : {1u, 4u}) {
            seam.positions.push_back(seam.positions[i]);
            seam.normals.push_back(seam.normals[i]);
            seam.uv.push_back(seam.uv[i]);
            seam.colors.push_back(seam.colors[i]);
            seam.tangents.push_back(seam.tangents[i]);
        }
        seam.indices[1] = 6;
        seam.indices[2] = 7;
        seam.colors[7] = {0, 0, 255, 77};
        auto split = seam;
        split.positions[5] = split.positions[4];
        split.indices.resize(12);
        const uint32_t seam_positions[] = {0, 1, 2, 3, 4, 5, 1, 4};
        CHECK(detail::merge_wedges(seam.view(), seam_positions, clusters, faces, split, nullptr));
        bool red = false, blue = false;
        for (size_t i = 0; i < split.positions.size(); ++i)
            if (length(split.positions[i]) < 1e-6) {
                auto c = split.colors[i];
                red |= c.r == 255 && c.b == 0;
                blue |= c.r == 0 && c.b == 255;
            }
        CHECK(red && blue);
        Mesh nonmanifold;
        nonmanifold.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}};
        nonmanifold.indices = {0, 1, 2, 1, 0, 3, 0, 1, 4};
        auto sheets = nonmanifold;
        const uint32_t three[] = {0, 1, 2};
        CHECK(detail::merge_wedges(nonmanifold.view(), ids, ids, three, sheets, nullptr));
        CHECK(sheets.positions.size() == 9);
        // A UV reversal restores the affected original fans and their exact UVs.
        auto folded = source;
        folded.exact_position_bits = {1u << 5};
        folded.positions[4] = folded.positions[5] = {2, 0, 0};
        folded.indices.resize(12);
        auto before = folded;
        CHECK(detail::merge_wedges(source.view(), positions, clusters, faces, folded, &stats));
        CHECK(stats.uv_rejections > 0 && folded.positions.size() == 6 &&
              uv_distortion(folded.view()).negative_uv_faces == 0);
        for (size_t c = 0; c < folded.indices.size(); ++c) {
            auto a = folded.uv[folded.indices[c]], b = before.uv[before.indices[c]];
            CHECK(a.x == b.x && a.y == b.y);
            CHECK(folded.view().exact_position(folded.indices[c]) ==
                  before.view().exact_position(before.indices[c]));
        }
        std::cout << "Continuous wedge storage, corner fans, discrete seams and fitting contracts "
                     "passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
