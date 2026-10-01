#include "training/json.hpp"
#include <iostream>

static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() {
    try {
        using blitz::NeuralActionStop;
        for (uint64_t target : {2u, 75u, 2048u}) {
            const auto shallow =
                policy_rollout_outcome(NeuralActionStop::TrialBudget, target + 1, target);
            require(shallow.at("version") == 1 && shallow.at("stop_reason") == "trial_budget" &&
                        shallow.at("target_triangles") == target && !shallow.at("target_reached"),
                    "budget-ended shallow rollout was reported as reaching its target");
            for (uint64_t actual : {target - 1, target}) {
                const auto reached =
                    policy_rollout_outcome(NeuralActionStop::TargetReached, actual, target);
                require(reached.at("target_reached") &&
                            reached.at("stop_reason") == "target_reached",
                        "reaching or crossing the requested triangle target was lost");
                const auto stopped =
                    policy_rollout_outcome(NeuralActionStop::Cancelled, actual, target);
                require(stopped.at("target_reached") && stopped.at("stop_reason") == "cancelled",
                        "cancellation hid actual target attainment");
            }
            const auto stopped =
                policy_rollout_outcome(NeuralActionStop::Cancelled, target + 1, target);
            const auto exhausted =
                policy_rollout_outcome(NeuralActionStop::NoAcceptedAction, target + 1, target);
            require(!stopped.at("target_reached") && !exhausted.at("target_reached") &&
                        exhausted.at("stop_reason") == "no_accepted_action",
                    "termination was mistaken for target attainment");
        }
        std::cout << "policy rollout outcome contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
