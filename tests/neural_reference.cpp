#include "evaluation_settings.hpp"
#include "neural/reference.hpp"
#include <array>
#include <iostream>

using namespace blitz;
static void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> static bool rejects(F&& action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
int main() try {
    Mesh source;
    source.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    source.indices = {0, 1, 2};
    neural::AuditReferences references(source.view());
    references.validate_view(source.view());
    Mesh predecessor = source;
    references.bind(predecessor.view());
    references.validate_view(predecessor.view());

    // A changed binding must be checked even when its allocation is reused.
    predecessor.indices[1] = 99;
    check(rejects([&] { references.bind(predecessor.view()); }),
          "invalid rebound indices accepted");
    check(rejects([&] { references.validate_view(predecessor.view()); }),
          "failed rebind retained trust in invalid storage");
    predecessor.indices[1] = 1;
    references.bind(predecessor.view());
    references.clear();
    predecessor.indices[1] = 98;
    check(rejects([&] { references.validate_view(predecessor.view()); }),
          "released reference retained trust after owner reuse");
    predecessor.indices[1] = 1;

    // Sharing source positions does not give an arbitrary topology trust.
    std::array<uint32_t, 3> indices{0, 1, 44};
    auto untrusted = source.view();
    untrusted.indices = indices;
    check(rejects([&] { references.validate_view(untrusted); }),
          "untrusted topology bypassed validation");
    Mesh invalid = source;
    invalid.positions[1].x = std::numeric_limits<float>::infinity();
    check(rejects([&] { references.validate_view(invalid.view()); }),
          "untrusted attributes bypassed validation");

    const auto bounds = blitz::bounds(source.view());
    for (auto profile : {Profile::Coverage, Profile::Normals, Profile::Attributes}) {
        EvalSettings settings;
        settings.profile = profile;
        for (double size : {16., 128., 512.}) {
            settings.screen_size = size;
            detail::validate_evaluation_settings(bounds, settings);
            check(evaluate(source.view(), source.view(), bounds, settings).passed,
                  "valid settings rejected");
        }
        settings.views = {0, 0, 1};
        check(rejects([&] { detail::validate_evaluation_settings(bounds, settings); }),
              "empty views accepted");
        settings.views = {1, 0, 1};
        settings.max_changed_area = -0.1;
        check(rejects([&] { detail::validate_evaluation_settings(bounds, settings); }),
              "invalid area accepted");
        settings.max_changed_area = 0.5;
        settings.screen_size = 16384;
        detail::validate_evaluation_settings(bounds, settings);
        for (double oversized : {16385., 1e100}) {
            settings.screen_size = oversized;
            check(rejects([&] { detail::validate_evaluation_settings(bounds, settings); }),
                  "oversized raster extent accepted before integer conversion");
        }
    }
    std::cout << "immutable audit reference contracts passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
