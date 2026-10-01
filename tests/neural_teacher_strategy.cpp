#include "training/teacher_labels.hpp"
#include "training/teacher_strategy.hpp"
#include <iostream>
#include <vector>
using namespace blitz::neural::training;
static void require(bool value, const char* why) {
    if (!value)
        throw std::runtime_error(why);
}
int main() {
    try {
        for (size_t count : {10, 11}) {
            CoverageTeacherSearch search(count);
            std::vector<uint8_t> visited;
            while (auto id = search.next()) {
                visited.push_back(*id);
                search.observe(*id, {.margin = .3, .valid = true, .known = true, .safe = true});
            }
            require(visited == (count == 10 ? std::vector<uint8_t>{0, 1, 2, 3}
                                            : std::vector<uint8_t>{0, 1, 2, 3, 10}),
                    "core omitted a placement or queried an offset");
            require(search.best() == 0 && !search.expanded(), "core tie ordering changed");
        }
        {
            CoverageTeacherSearch search(11);
            while (auto id = search.next())
                search.observe(*id, {.margin = 2, .valid = true, .known = true});
            require(!search.safe(*search.best()) && search.expand(),
                    "failed core did not permit fallback");
            while (auto id = search.next())
                search.observe(*id, {.margin = .4, .valid = true, .known = true, .safe = true});
            require(search.best() == 4 && !search.expand(),
                    "safe offset did not replace failed core");
        }
        {
            // A false-positive bound won and pruned the only valid replacement.
            // Exact rejection must revisit that replacement and all six offsets.
            CoverageTeacherSearch search(11);
            search.observe(0, {.margin = .1, .valid = true, .known = true, .safe = true});
            search.observe(1, {.valid = true, .pruned = true});
            for (auto id : {2, 3, 10})
                search.observe(uint8_t(id), {.valid = true, .known = true});
            require(search.reject(0) == (1u << 1), "incumbent rejection kept a stale prune");
            require(search.next() == 1, "pruned core alternative was not retried");
            search.observe(1, {.margin = .2, .valid = true, .known = true, .safe = true});
            std::vector<uint8_t> offsets;
            while (auto id = search.next()) {
                offsets.push_back(*id);
                search.observe(*id, {.valid = true, .known = true});
            }
            require(offsets == std::vector<uint8_t>{4, 5, 6, 7, 8, 9} && search.best() == 1,
                    "fallback did not search the complete union");
            require(search.rejected() == 1 && search.queried() == 0x7ff,
                    "query/rejection provenance lost");
            search.reject(1);
            require(search.best() == 2 && !search.safe(*search.best()),
                    "exactly rejected candidate became safe again");
        }
        {
            CoverageTeacherSearch search(10);
            search.observe(0, {.margin = 0, .valid = true, .known = true, .safe = true});
            require(!search.next(), "zero bound should stop redundant queries");
            search.reject(0);
            require(search.next() == 1 && search.expanded(),
                    "zero-bound rejection lost unqueried candidates");
            while (auto id = search.next())
                search.observe(*id, {.margin = .5, .valid = true, .known = true, .safe = true});
            for (uint8_t id = 1; id < 10; ++id) {
                require(search.best() == id, "rejected finalist repeated or replacement skipped");
                search.reject(id);
            }
            require(!search.best() && !search.next() && search.rejected() == 0x3ff,
                    "rejection exhaustion did not terminate");
        }
        {
            CoverageTeacherSearch search(10);
            search.observe(0, {.valid = true});
            require(search.incomplete() && !search.next() && !search.best(),
                    "unknown audit became a known negative");
            CoverageTeacherSearch invalid(10);
            invalid.observe(0, {});
            require(!invalid.incomplete() && invalid.next() == 1,
                    "invalid geometry made a completed audit unknown");
            CoverageTeacherSearch partial(11);
            partial.observe(0, {.margin = .2, .valid = true, .known = true, .safe = true});
            partial.observe(1, {.valid = true});
            require(partial.incomplete() && !partial.next(),
                    "prior safe bound hid an unknown audit");
        }
        {
            // Per-edge early exit must not suppress other queried edges with an
            // independently confirmed equivalent triangle count and margin.
            std::array<TeacherChoice, 4> choices{
                {{8, .2, true}, {8, .2, true}, {8, .1, false}, {10, 0, true}}};
            require(preferred_actions(choices) == 3, "equivalent queried actions lost their tie");
            choices[1].confirmed = false;
            require(preferred_actions(choices) == 1, "unknown action acquired a preference");
        }
        for (auto name : {"exhaustive", "coverage-core-first"})
            require(teacher_strategy_name(teacher_strategy_option(name)) == name,
                    "strategy contract does not round trip");
        bool rejected = false;
        try {
            teacher_strategy_option("fast");
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "unknown strategy silently changed the experiment");
        std::cout << "teacher strategy contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
