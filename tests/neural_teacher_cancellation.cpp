#include "training/teacher_cancellation.hpp"
#include <iostream>
#include <utility>
using namespace blitz::neural;
using namespace blitz::neural::training;
static void require(bool value, const char* why) {
    if (!value)
        throw std::runtime_error(why);
}
int main() {
    try {
        // Arm the event explicitly instead of depending on the number of audit
        // polls. The callback stops reporting it as soon as it is consumed.
        bool event = false;
        auto one_shot = [&] { return std::exchange(event, false); };
        TeacherCancellation preparation;
        require(!preparation.poll(one_shot), "fresh preparation started cancelled");
        event = true;
        require(preparation.poll(one_shot) && !event, "one-shot cancellation was not observed");
        require(preparation.poll(one_shot), "preparation forgot an observed stop");
        bool repolled = false;
        require(preparation.poll([&] {
            repolled = true;
            return false;
        }) && !repolled,
                "stopped preparation queried the external callback again");

        for (size_t count : {size_t(1), size_t(4), size_t(8)}) {
            for (size_t lane : {size_t(0), count - 1}) {
                for (bool valid : {false, true}) {
                    for (bool pruned : {false, true}) {
                        TeacherCancellation stopped;
                        std::array<CandidateAudit, 8> batch{};
                        // Vulkan can stop before a lane's first camera, leaving
                        // valid=false. Previously that event became bad geometry.
                        batch[lane].valid = valid;
                        batch[lane].pruned = pruned;
                        batch[lane].value.cancelled = true;
                        stopped.observe(std::span(batch).first(count));
                        require(stopped.stopped() && stopped.poll([] { return false; }),
                                "batch validity or pruning hid cancellation");
                        batch[lane].value.cancelled = false;
                        stopped.observe(std::span(batch).first(count));
                        require(stopped.stopped(), "later batch cleared an observed stop");
                    }
                }
            }
        }
        TeacherCancellation running;
        std::array<CandidateAudit, 3> results{};
        results[1].valid = true;
        results[1].pruned = true;
        results[2].valid = true;
        results[2].value.verdict = AuditVerdict::Fail;
        running.observe(results);
        require(!running.stopped() && !running.poll([] { return false; }),
                "invalid, pruned or failed geometry became cancellation");
        running.observe({});
        require(!running.stopped(), "empty batch became cancellation");
        std::cout << "teacher cancellation contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
