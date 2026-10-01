#include "chain_hooks.hpp"
#include <iostream>

using namespace blitz;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() try {
    Mesh source;
    source.positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    source.indices = {0, 1, 2, 0, 2, 3};
    Settings settings;
    settings.levels = 2;
    settings.base_pixels = 16;
    settings.last_pixels = 8;
    settings.profile = Profile::Coverage;
    settings.candidate_budget = 1;
    settings.beam_width = 1;
    settings.research.trace = true;
    detail::GenerationHooks hooks;
    hooks.evaluate = [](MeshView, MeshView, const Bounds&, const EvalSettings&) {
        return Measurement{};
    };
    hooks.confirm = [](Result&) { return true; };
    unsigned calls = 0;
    hooks.propose_guarded = [&](MeshView input, MeshView fixed, MeshView previous, const Bounds&,
                                const ReduceSettings&, const EvalSettings&, const EvalSettings&,
                                const EvalSettings&, const EvalSettings&) {
        require(same_mesh_data(fixed, source.view()), "source reference changed during hook merge");
        require(same_mesh_data(previous, source.view()), "wrong first predecessor");
        ++calls;
        Lod proposal;
        proposal.shared_vertices = false;
        proposal.data = copy_mesh(input);
        proposal.data.indices.resize(3);
        return proposal;
    };
    for (uint32_t budget : {0u, 10000u}) {
        calls = 0;
        settings.max_added_vertex_bytes_bps = budget;
        auto result = detail::generate_with_hooks(source.view(), settings, {}, &hooks);
        require(result.status == Status::Complete && result.lods.size() == 2,
                "hook chain incomplete");
        require(calls == 1 && result.candidate_evaluations == 1,
                "classical retry consumed hook work budget");
        require(result.lods.back().view(source.view()).triangles() == (budget ? 1u : 2u),
                "hook ignored vertex cap");
        require(result.added_vertex_budget_bytes &&
                    storage_stats(result).added_vertex_bytes <= *result.added_vertex_budget_bytes,
                "hook storage exceeds cap");
        require(result.vertex_budget_rejections == (budget ? 0u : 1u),
                "hook budget rejection disappeared");
        require(result.tail_probe_evaluations == 0 && !result.adaptive_retry_attempted,
                "classical search ran around hooks");
        require(result.audit.profile == settings.profile &&
                    result.audit.audit_views.orthographic == settings.audit_views.orthographic,
                "hook lost exported audit contract");
    }
    // Packed rendering can reject the raw source while an owned baseline is
    // valid. A cap that excludes every owned option must retain only LOD0.
    hooks.source_fallback_requires_audit = true;
    hooks.evaluate = [&](MeshView, MeshView candidate, const Bounds&, const EvalSettings&) {
        Measurement m;
        if (same_mesh_data(candidate, source.view())) {
            m.complete = m.passed = false;
            m.error = 100;
        }
        return m;
    };
    hooks.fallback = [&](MeshView, const EvalSettings&, const EvalSettings&, const EvalSettings&,
                         const EvalSettings&) -> std::optional<Lod> {
        Lod baseline;
        baseline.shared_vertices = false;
        baseline.data = source;
        baseline.data.positions[0].z = .01f;
        return baseline;
    };
    for (uint32_t budget : {0u, 10000u}) {
        settings.max_added_vertex_bytes_bps = budget;
        auto result = detail::generate_with_hooks(source.view(), settings, {}, &hooks);
        require(result.status == (budget ? Status::Complete : Status::BudgetLimited),
                "owned fallback cap did not report its limiting constraint");
        require(result.lods.size() == (budget ? 2u : 1u),
                "owned fallback cap published an unaudited level");
        require(same_mesh_data(result.lods.front().view(source.view()), source.view()),
                "owned fallback cap changed LOD0");
        require(storage_stats(result).added_vertex_bytes <= *result.added_vertex_budget_bytes,
                "owned fallback exceeded the explicit vertex cap");
        if (budget)
            require(result.lods.back().view(source.view()).triangles() == 1,
                    "sufficient vertex budget discarded the audited reduction");
        else
            require(result.vertex_budget_rejections > 0,
                    "infeasible cap did not retain rejection diagnostics");
    }
    settings.max_added_vertex_bytes_bps.reset();
    hooks.evaluate = [](MeshView, MeshView, const Bounds&, const EvalSettings&) {
        Measurement m;
        m.complete = m.passed = false;
        m.error = 100;
        return m;
    };
    hooks.fallback = [](MeshView, const EvalSettings&, const EvalSettings&, const EvalSettings&,
                        const EvalSettings&) -> std::optional<Lod> { return {}; };
    bool infeasible = false;
    try {
        detail::generate_with_hooks(source.view(), settings, {}, &hooks);
    } catch (const std::logic_error& e) {
        infeasible = std::string_view(e.what()) == "no audited chain candidate or fallback";
    }
    require(infeasible, "representation failure was mislabeled as a vertex budget limit");
    hooks.fallback = {};
    // A single observed cancellation remains a cancellation after callbacks
    // return false again, including an apparent pixel failure in that sample.
    hooks.source_fallback_requires_audit = true;
    bool first = true;
    hooks.evaluate = [&](MeshView, MeshView, const Bounds&, const EvalSettings&) {
        Measurement m;
        if (first) {
            first = false;
            m.cancelled = true;
            m.complete = m.passed = false;
            m.error = 100;
        }
        return m;
    };
    calls = 0;
    settings.cancelled = [] { return false; };
    auto cancelled = detail::generate_with_hooks(source.view(), settings, {}, &hooks);
    require(cancelled.status == Status::Cancelled && cancelled.lods.size() == 1 && calls == 0,
            "latched audit cancellation published unconfirmed hook levels");
    for (size_t prefix : {31u, 32u, 33u})
        for (bool source_bitmap : {false, true}) {
            Mesh original;
            original.positions.resize(prefix);
            if (source_bitmap)
                original.exact_position_bits.resize((prefix + 31) / 32);
            auto pool = std::make_shared<Mesh>();
            pool->positions.resize(prefix + 2);
            pool->exact_position_bits.resize((prefix + 2 + 31) / 32);
            Lod lod;
            lod.shared_vertices = false;
            lod.vertex_pool = pool;
            lod.source_prefix_vertices = uint32_t(prefix);
            const uint64_t bitmap =
                pool->exact_position_bits.size() * 4 - original.exact_position_bits.size() * 4;
            require(added_vertex_bytes(lod, original.view()) == 24 + bitmap,
                    "shared-prefix bitmap rounded into per-vertex stride");
        }
    std::cout << "merged neural chain contracts passed\n";
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
