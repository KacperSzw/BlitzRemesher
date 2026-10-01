#include "training/teacher_seed.hpp"
#include <iostream>
#include <stdexcept>

using namespace blitz;
using namespace blitz::neural::training;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() {
    try {
        for (const auto limits : {std::array{3., 2.}, std::array{1.5, 4.}, std::array{2., 2.}}) {
            EvalSettings source;
            source.limit = limits[0];
            source.screen_size = 128;
            source.views = {7, 3, 19};
            source.max_changed_area = .25;
            const auto packed = teacher_packing_settings(source, limits[1]);
            require(packed.limit <= limits[0] && packed.limit <= limits[1] &&
                        (packed.limit == limits[0] || packed.limit == limits[1]),
                    "packing admission does not enforce both original-source limits");
            require(packed.screen_size == 128 && packed.views.rotation_seed == 19 &&
                        packed.views.orthographic == 7 && packed.views.perspective == 3 &&
                        packed.max_changed_area == .25 && source.limit == limits[0],
                    "packing changed camera, area or caller settings");

            for (bool predecessor : {false, true}) {
                auto destination = source;
                destination.screen_size = 64;
                for (double error : {1., 2., 2.5, 4.5}) {
                    std::vector<std::pair<double, double>> calls;
                    const auto result =
                        audit_teacher_seed(source, limits[1], predecessor ? &destination : nullptr,
                                           [&](const EvalSettings& e) {
                                               calls.emplace_back(e.screen_size, e.limit);
                                               Measurement m;
                                               m.error = error;
                                               m.passed = error <= e.limit;
                                               return m;
                                           });
                    require(result.passed() == (error <= limits[0] && error <= limits[1]),
                            "a seed passed only one of source and adjacent admission");
                    require(calls.size() == 1 + size_t(limits[0] != limits[1]) + predecessor,
                            "equal-limit reuse removed an independent destination audit");
                    require(calls.front() == std::pair{128., limits[0]},
                            "source admission used different cameras or threshold");
                    if (limits[0] != limits[1])
                        require(calls[1] == std::pair{128., limits[1]},
                                "adjacent admission was not against the current source screen");
                    if (predecessor)
                        require(calls.back() == std::pair{64., limits[0]},
                                "predecessor seed omitted the destination screen");
                }
                const auto failed_destination =
                    audit_teacher_seed(source, limits[1], predecessor ? &destination : nullptr,
                                       [](const EvalSettings& e) {
                                           Measurement m;
                                           m.passed = e.screen_size != 64;
                                           return m;
                                       });
                require(failed_destination.passed() != predecessor,
                        "destination failure was admitted as an initial state");
            }
        }
        // A seed cannot become an incumbent on an unknown/cancelled audit,
        // even if a caller accidentally leaves the default passed flag set.
        for (size_t gate = 0; gate < 3; ++gate)
            for (unsigned failure = 0; failure < 5; ++failure) {
                TeacherSeedAudit audit;
                auto* m = std::array{&audit.source, &audit.adjacent, &audit.destination}[gate];
                switch (failure) {
                case 0:
                    m->complete = false;
                    break;
                case 1:
                    m->cancelled = true;
                    break;
                case 2:
                    m->resource_limited = true;
                    break;
                case 3:
                    m->error = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 4:
                    m->changed_area = std::numeric_limits<double>::infinity();
                    break;
                }
                require(!audit.passed(), "unknown seed audit became an accepted initial state");
            }
        std::cout << "teacher seed admission contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
