#include "chain_hooks.hpp"
#include <iostream>

using namespace blitz;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static void proposal_budget_contract() {
    Mesh source;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
            source.positions.push_back({float(x), float(y), 0});
    for (uint32_t y = 0; y < 2; ++y)
        for (uint32_t x = 0; x < 2; ++x) {
            const uint32_t a = y * 3 + x;
            source.indices.insert(source.indices.end(), {a, a + 1, a + 4, a, a + 4, a + 3});
        }
    Settings settings;
    settings.levels = 4;
    settings.base_pixels = 32;
    settings.last_pixels = 8;
    settings.profile = Profile::Coverage;
    settings.triangle_overhead_bps = 0;
    settings.beam_width = 1;
    settings.research.trace = true;
    detail::GenerationHooks hooks;
    // This fixture tests search allocation; its synthetic evaluator makes no
    // visual-quality claim. Only endpoint proposals can reduce the fixture.
    hooks.evaluate = [](MeshView, MeshView, const Bounds&, const EvalSettings&) {
        return Measurement{};
    };
    hooks.confirm = [](Result&) { return true; };
    hooks.propose_guarded = [](MeshView input, MeshView, MeshView, const Bounds&,
                               const ReduceSettings& rs, const EvalSettings&, const EvalSettings&,
                               const EvalSettings&, const EvalSettings&) {
        Lod proposal;
        proposal.shared_vertices = rs.output == OutputMode::Reuse;
        proposal.data = copy_mesh(input);
        if (proposal.shared_vertices)
            proposal.data.indices.resize(rs.target_triangles * 3);
        return proposal;
    };
    for (uint16_t budget : {2, 4}) {
        settings.candidate_budget = budget;
        auto result = detail::generate_with_hooks(source.view(), settings, {}, &hooks);
        require(result.status == Status::Complete && result.lods.size() == 4,
                "bounded hook schedule incomplete");
        if (budget == 2)
            for (size_t level = 0; level < result.lods.size(); ++level)
                require(result.lods[level].view(source.view()).triangles() == (8u >> level),
                        "small proposal budget starved endpoint reduction at a later LOD");
        require(result.lods.back().view(source.view()).triangles() == 1,
                "extra proposal budget failed to reach the fixture's reduction bound");
        for (uint8_t level = 1; level < 4; ++level) {
            std::vector<ProposalTrace> proposals;
            for (const auto& proposal : result.proposals)
                if (proposal.level == level)
                    proposals.push_back(proposal);
            require(proposals.size() == budget, "hook proposal budget changed");
            require(proposals[0].strategy != proposals[1].strategy,
                    "first two proposals must cover both output modes");
            require(level == 1 || proposals[0].origin != proposals[1].origin,
                    "first two proposals must cover both distinct origins");
        }
    }
    for (auto output : {OutputMode::Reuse, OutputMode::Rebuild})
        for (auto chain : {ChainMode::Direct, ChainMode::Progressive, ChainMode::Hybrid}) {
            settings.candidate_budget = 2;
            settings.research.output = output;
            settings.research.chain = chain;
            auto result = detail::generate_with_hooks(source.view(), settings, {}, &hooks);
            require(result.status == Status::Complete, "forced hook schedule incomplete");
            for (const auto& proposal : result.proposals) {
                require(proposal.strategy == (output == OutputMode::Reuse ? 4 : 5),
                        "hook schedule ignored forced output mode");
                if (chain != ChainMode::Hybrid)
                    require(proposal.origin == (chain == ChainMode::Direct ? 0 : 1),
                            "hook schedule ignored forced origin");
            }
        }
}
int main() try {
    proposal_budget_contract();
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
