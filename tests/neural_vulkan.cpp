#include "neural/action_gpu.hpp"
#include "neural/cuda.cuh"
#include "neural/internal.hpp"
#include "neural/raster_pixel.cuh"
#include "neural/teardown_trace.hpp"
#include "neural/vulkan.hpp"
#include "neural/vulkan_cuda.hpp"
#include "training/worker_retirement.hpp"
#include <barrier>
#include <future>
#include <iostream>
#include <mutex>
using namespace blitz;
using namespace blitz::neural;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static Mesh fixture() {
    Mesh m;
    m.positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    m.normals.assign(4, {0, 0, 1});
    m.uv = {{-8, -8}, {8, -8}, {8, 8}, {-8, 8}};
    m.tangents.assign(4, {1, 0, 0, -1});
    m.colors.assign(4, {31, 127, 255, 47});
    m.indices = {0, 1, 2, 0, 2, 3};
    m.materials = {65535, 65535};
    return m;
}
static void draw_domain_contracts(bool progress = false) {
    auto mesh = fixture();
    auto box = bounds(mesh.view());
    NeuralOptions options;
    options.memory_mib = 128;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    gpu::Device device(options);
    GpuActionState state(mesh.view(), options, true);
    VulkanRaster raster(options);
    auto view = state.view();
    auto positions = gpu::upload_stream(device, mesh.view().positions);
    view.positions = positions.p;
    gpu::Buffer<AuditPixel> pixels(device, 64 * 64);
    Camera camera{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, box.radius * 4, 1, 24 / box.diameter(), false};
    unsigned draw_id = 0;
    auto draw = [&] {
        if (progress)
            std::cout << "domain draw " << ++draw_id << std::endl;
        raster.render(view, box, camera, 24, 2, false, NeuralVertexStorage::Packed, pixels.p,
                      nullptr);
    };
    auto reject = [&] {
        bool rejected = false;
        try {
            draw();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "malformed packed device stream escaped validation");
        VertexBounds domain{view.quant_low, view.quant_extent};
        if (view.fixed_quantization && !valid_vertex_bounds(domain)) {
            rejected = false;
            try {
                GpuActionState bad(mesh.view(), options, true, &domain);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "malformed fixed domain reached working mesh encoding");
        }
    };
    for (bool fixed : {false, true})
        for (float value : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
            auto bad = mesh.positions;
            bad[0].x = value;
            positions.upload(bad);
            view.fixed_quantization = fixed;
            ++view.revision;
            reject();
        }
    positions.upload(mesh.positions);
    view.fixed_quantization = true;
    auto domain = vertex_bounds(mesh.view());
    for (float value : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        view.quant_low = domain.low;
        view.quant_extent = domain.extent;
        view.quant_low.x = value;
        ++view.revision;
        reject();
    }
    for (float value : {-1.f, INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        view.quant_low = domain.low;
        view.quant_extent = domain.extent;
        view.quant_extent.x = value;
        ++view.revision;
        reject();
    }
    view.quant_low = {std::numeric_limits<float>::max(), 0, 0};
    view.quant_extent = view.quant_low;
    ++view.revision;
    reject();
    view.quant_low = domain.low;
    view.quant_extent = domain.extent;
    ++view.revision;
    draw();
    unsigned covered = 0;
    for (const auto& p : pixels.download())
        covered += p.covered;
    require(covered > 100, "valid draw failed after malformed cached draws");
}
static Mesh candidate_fixture() {
    Mesh mesh;
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) {
            mesh.positions.push_back({float(x), float(y), 0});
            mesh.normals.push_back({0, 0, 1});
        }
    for (uint32_t y = 0; y < 3; ++y)
        for (uint32_t x = 0; x < 3; ++x) {
            auto i = y * 4 + x;
            mesh.indices.insert(mesh.indices.end(), {i, i + 1, i + 4, i + 1, i + 5, i + 4});
        }
    mesh.double_sided = {1};
    return mesh;
}
static void candidate_contracts(bool memory_boundaries = false) {
    auto mesh = candidate_fixture();
    NeuralOptions options;
    options.memory_mib = 256;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    options.view_batch = 1;
    AuditSession session(options);
    GpuActionState state(mesh.view(), options, true);
    AuditCuda audit(options, mesh.view());
    auto rows = state.teacher_actions({}, 1, 917);
    require(!rows.empty(), "candidate fixture has no legal edits");
    auto alternatives = state.teacher_proposals(rows[0].action);
    std::array<GpuActionState::Proposal, 8> proposals{
        alternatives[0], alternatives[2], alternatives[4], alternatives[4],
        alternatives[0], alternatives[2], alternatives[4], alternatives[4]};
    proposals[3].placement.position.x = INFINITY;
    proposals[7].placement.position.y = INFINITY;
    EvalSettings e;
    e.screen_size = 24;
    e.limit = 3;
    e.views = {3, 1, 817};
    e.supersample = 2;
    e.max_supersample = 4;
    e.max_changed_area = .4;
    auto box = bounds(mesh.view());
    for (auto profile : {Profile::Coverage, Profile::Attributes})
        for (auto count : {1u, 2u, 4u, 8u})
            for (double cutoff : {std::numeric_limits<double>::infinity(), 0., .02}) {
                // Instrument the maximum lane count with valid/invalid candidates and
                // pruning in both attachment layouts. CTest retains the full matrix.
                if (memory_boundaries && (count != 8 || cutoff != .02))
                    continue;
                e.profile = profile;
                std::array<CandidateAudit, 8> expected;
                for (unsigned i = 0; i < count; ++i) {
                    DeviceMeshView candidate;
                    auto& q = expected[i];
                    q.valid = state.trial(rows[0].action, proposals[i].placement, candidate);
                    if (q.valid) {
                        q.faces = candidate.faces;
                        q.value = audit.certify(mesh.view(), candidate, box, e, nullptr, cutoff,
                                                std::isfinite(cutoff) ? &q.pruned : nullptr);
                    }
                }
                auto views = state.trial_batch(rows[0].action, std::span(proposals).first(count));
                auto actual = audit.certify_candidates(mesh.view(), views, box, e, nullptr, cutoff);
                for (unsigned i = 0; i < count; ++i) {
                    auto& a = actual[i];
                    auto& b = expected[i];
                    require(a.valid == b.valid, "GPU indirect validity differs");
                    if (!a.valid)
                        continue;
                    require(a.faces == b.faces && a.pruned == b.pruned &&
                                a.value.verdict == b.value.verdict &&
                                a.value.views == b.value.views &&
                                a.value.changed_area == b.value.changed_area &&
                                a.value.error_upper == b.value.error_upper,
                            "candidate batch differs from serial audit");
                }
            }
    require(state.view().faces == mesh.view().triangles(), "speculative batch committed geometry");
}
static void same_measurement(const Measurement& a, const Measurement& b) {
    require(std::tie(a.error, a.coverage, a.coverage_upper, a.changed_area, a.normal_degrees,
                     a.worst_view, a.changed_area_worst_view, a.views_evaluated, a.supersample,
                     a.complete, a.passed, a.resource_limited, a.cancelled) ==
                std::tie(b.error, b.coverage, b.coverage_upper, b.changed_area, b.normal_degrees,
                         b.worst_view, b.changed_area_worst_view, b.views_evaluated, b.supersample,
                         b.complete, b.passed, b.resource_limited, b.cancelled),
            "grouped exact audit changed metrics or completion");
}
static void same_candidates(std::span<const CandidateAudit> a, std::span<const CandidateAudit> b) {
    require(a.size() == b.size(), "grouped candidate count differs");
    for (size_t i = 0; i < a.size(); ++i) {
        auto& x = a[i];
        auto& y = b[i];
        require(std::tie(x.faces, x.valid, x.pruned, x.value.verdict, x.value.error_upper,
                         x.value.changed_area, x.value.views, x.value.supersample,
                         x.value.resource_limited, x.value.cancelled) ==
                    std::tie(y.faces, y.valid, y.pruned, y.value.verdict, y.value.error_upper,
                             y.value.changed_area, y.value.views, y.value.supersample,
                             y.value.resource_limited, y.value.cancelled),
                "grouped candidate audit changed labels, lane identity, or pruning");
    }
}
static void grouped_candidate_contracts(bool memory_boundaries) {
    auto mesh = candidate_fixture(), previous = mesh;
    previous.positions[5].x += .17f;
    auto box = bounds(mesh.view());
    NeuralOptions options;
    options.memory_mib = 128;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    options.vertex_storage = NeuralVertexStorage::Packed;
    options.mask_only_coverage = true;
    GpuActionState state(mesh.view(), options, true);
    auto rows = state.teacher_actions({}, 1, 919);
    require(!rows.empty(), "grouped fixture has no legal edits");
    auto alternatives = state.teacher_proposals(rows[0].action);
    std::array<GpuActionState::Proposal, 4> proposals{alternatives[0], alternatives[2],
                                                      alternatives[4], alternatives[6]};
    proposals[3].placement.position.x = INFINITY;
    EvalSettings source;
    source.profile = Profile::Coverage;
    source.screen_size = 24;
    source.limit = 3;
    source.views = {3, 1, 991};
    source.supersample = 2;
    source.max_supersample = 4;
    source.max_changed_area = .4;
    auto adjacent = source;
    adjacent.limit = 1.5;
    auto destination = source;
    destination.screen_size = 16;
    for (uint8_t batch : {1, 4})
        for (bool direct : {false, true}) {
            if (memory_boundaries && (batch != 4 || !direct))
                continue;
            options.view_batch = batch;
            options.direct_targets = direct;
            auto uncached = options;
            uncached.cache_rasters = false;
            AuditCuda control(uncached, mesh.view()), tested(options, mesh.view());
            control.bind_reference(previous.view());
            tested.bind_reference(previous.view());
            for (double cutoff : {std::numeric_limits<double>::infinity(), .02}) {
                auto views = state.trial_batch(rows[0].action, proposals);
                // The second reference receives a non-prefix, reordered subset.
                // A cache addressed by its compacted lane index would be wrong.
                std::array<DeviceMeshView, 2> subset{views[2], views[0]};
                auto expected_source =
                    control.certify_candidates(mesh.view(), views, box, source, nullptr, cutoff);
                auto expected_adjacent = control.certify_candidates(previous.view(), subset, box,
                                                                    adjacent, nullptr, cutoff);
                NeuralStats stats;
                tested.with_candidate_rasters(views, [&] {
                    same_candidates(
                        tested.certify_candidates(mesh.view(), views, box, source, &stats, cutoff),
                        expected_source);
                    same_candidates(tested.certify_candidates(previous.view(), subset, box,
                                                              adjacent, &stats, cutoff),
                                    expected_adjacent);
                });
                require(stats.gpu_candidate_render_hits > 0,
                        "grouped candidate masks were never reused");
                require(!expected_source[3].valid, "invalid lane was not exercised");
            }
            // A refined image can be the next audit's initial sampling. Its
            // cached draw count must be the live count, never the trial capacity.
            auto views = state.trial_batch(rows[0].action, proposals);
            auto coarse = source;
            coarse.limit = .4;
            auto fine = source;
            fine.supersample = fine.max_supersample = 4;
            auto expected_fine = control.certify_candidates(mesh.view(), views, box, fine);
            NeuralStats refinement_stats;
            tested.with_candidate_rasters(views, [&] {
                tested.certify_candidates(mesh.view(), views, box, coarse, &refinement_stats);
                same_candidates(
                    tested.certify_candidates(mesh.view(), views, box, fine, &refinement_stats),
                    expected_fine);
            });
            require(refinement_stats.gpu_candidate_render_hits > 0,
                    "refined candidate images were not reused at their exact sampling");
            // Identical pointers are reused by every serial trial; scopes must not
            // retain a preceding placement, including after an invalid attempt.
            for (size_t i : {0u, 2u}) {
                DeviceMeshView candidate;
                require(state.trial(rows[0].action, proposals[i].placement, candidate),
                        "grouped exact fixture placement invalid");
                std::array<Measurement, 4> expected{
                    control.evaluate(mesh.view(), candidate, box, source),
                    control.evaluate(previous.view(), candidate, box, adjacent),
                    control.evaluate(mesh.view(), candidate, box, destination),
                    control.evaluate(mesh.view(), candidate, box, source)};
                NeuralStats stats;
                tested.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                    same_measurement(tested.evaluate(mesh.view(), candidate, box, source, &stats),
                                     expected[0]);
                    same_measurement(
                        tested.evaluate(previous.view(), candidate, box, adjacent, &stats),
                        expected[1]);
                    same_measurement(
                        tested.evaluate(mesh.view(), candidate, box, destination, &stats),
                        expected[2]);
                    same_measurement(tested.evaluate(mesh.view(), candidate, box, source, &stats),
                                     expected[3]);
                });
                require(stats.gpu_candidate_render_hits > 0,
                        "grouped exact candidate mask was never reused");
                require(stats.gpu_reference_render_hits > 0,
                        "source resolution switch discarded both reference domains");
                require(!state.trial(rows[0].action, proposals[3].placement, candidate),
                        "invalid intervening trial was not exercised");
            }
            // Callback exceptions release the group even after it retained images.
            auto candidate = state.view();
            bool propagated = false;
            try {
                tested.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                    tested.evaluate(mesh.view(), candidate, box, source);
                    throw std::runtime_error("test-owned interruption");
                });
            } catch (const std::runtime_error&) {
                propagated = true;
            }
            require(propagated, "group swallowed callback exception");
            bool nested = false;
            tested.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                try {
                    tested.with_candidate_rasters(std::span{&candidate, 1}, [] {});
                } catch (const std::invalid_argument&) {
                    nested = true;
                }
                same_measurement(tested.evaluate(mesh.view(), candidate, box, source),
                                 control.evaluate(mesh.view(), candidate, box, source));
            });
            require(nested, "nested raster group was accepted");
            // Cancellation during a query remains unknown with identical polling.
            uint32_t polls = 0;
            auto cancelled = source;
            cancelled.cancelled = [&] { return ++polls == 3; };
            auto expected = control.certify(mesh.view(), candidate, box, cancelled);
            const auto expected_polls = polls;
            polls = 0;
            tested.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                // Populate masks without polling first, then cancel while the
                // second audit reads those masks instead of drawing again.
                tested.certify(mesh.view(), candidate, box, source);
                NeuralStats hits;
                auto actual = tested.certify(mesh.view(), candidate, box, cancelled, &hits);
                require(actual.verdict == expected.verdict &&
                            actual.cancelled == expected.cancelled &&
                            actual.views == expected.views && polls == expected_polls,
                        "group changed cancellation polling or made an unknown label known");
                require(hits.gpu_candidate_render_hits > 0,
                        "cancellation fixture did not exercise a cached candidate");
            });
            auto attributes = source;
            attributes.profile = Profile::Attributes;
            tested.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                tested.evaluate(mesh.view(), candidate, box, source);
                same_measurement(tested.evaluate(mesh.view(), candidate, box, attributes),
                                 control.evaluate(mesh.view(), candidate, box, attributes));
            });
            // One query exceeds workspace, the other the sample cap. Both remain
            // resource results and release optional masks before a bounded retry.
            for (double screen : {1024., 4096.}) {
                auto unavailable = source;
                unavailable.screen_size = screen;
                unavailable.supersample = unavailable.max_supersample = 4;
                auto failure = control.evaluate(mesh.view(), candidate, box, unavailable);
                require(failure.resource_limited && !failure.complete,
                        "resource fixture did not fail");
                tested.with_candidate_rasters(std::span{&candidate, 1}, [&] {
                    same_measurement(tested.evaluate(mesh.view(), candidate, box, unavailable),
                                     failure);
                    same_measurement(tested.evaluate(mesh.view(), candidate, box, source),
                                     control.evaluate(mesh.view(), candidate, box, source));
                });
            }
        }
    // A fresh evaluator makes the second-screen retention assertion independent
    // of preceding group hits. Only the third call can reuse a source image.
    AuditCuda domains(options, mesh.view());
    NeuralStats stats;
    auto candidate = state.view();
    auto first = domains.evaluate(mesh.view(), candidate, box, source, &stats);
    domains.evaluate(mesh.view(), candidate, box, destination, &stats);
    require(stats.gpu_reference_render_hits == 0, "source domain fixture was already cached");
    same_measurement(domains.evaluate(mesh.view(), candidate, box, source, &stats), first);
    require(first.complete && stats.gpu_reference_render_hits == first.views_evaluated,
            "switching source resolution discarded the previous source images");
    require(state.view().faces == mesh.view().triangles(), "group committed a trial");
}
void packed_seed_domain() {
    auto source = fixture(), seed = source, original = source;
    auto q = vertex_bounds(source.view());
    for (auto& p : seed.positions) {
        if (p.x == -1)
            p.x = -1.25f;
        else if (p.x == 1)
            p.x = 1.5f;
    }
    auto proposal = seed;
    NeuralOptions options;
    options.memory_mib = 128;
    options.vertex_storage = NeuralVertexStorage::Packed;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    GpuActionState state(seed.view(), options, true, &q, source.view());
    auto working = state.snapshot().data;
    require(working.view().triangles() == source.view().triangles() &&
                state.view().fixed_quantization,
            "packed episode lost topology or its fixed domain");
    for (const auto& p : working.positions)
        require(p.x >= -1 && p.x <= 1 && p.y >= -1 && p.y <= 1 && p.z == 0,
                "packed seed escaped source bounds");
    EvalSettings e;
    e.profile = Profile::Coverage;
    e.screen_size = 16;
    e.limit = 2;
    e.supersample = 2;
    e.max_supersample = 4;
    e.views = {2, 1, 819};
    AuditCuda audit(options, source.view());
    auto measured = audit.evaluate(source.view(), state.view(), bounds(source.view()), e);
    require(measured.complete && measured.passed,
            "supported packed seed failed unchanged source audit");
    require(same_mesh_data(source.view(), original.view()) &&
                same_mesh_data(seed.view(), proposal.view()),
            "seed preparation changed supplied streams");
    seed.uv[0].x = 8.01f;
    bool rejected = false;
    try {
        GpuActionState invalid(seed.view(), options, true, &q, source.view());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "packed seed silently clamped an unsupported UV");
}
static void mask_memory_contracts() {
    auto mesh = fixture();
    auto box = bounds(mesh.view());
    NeuralOptions options;
    options.memory_mib = 128;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    MemoryScope memory(options);
    gpu::Device device(options);
    GpuActionState state(mesh.view(), options, true);
    VulkanRaster raster(options);
    gpu::Buffer<AuditPixel> pixels(device, 64 * 64);
    gpu::Buffer<uint32_t> words(device, 64 * 64 / 32);
    Camera camera{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, box.radius * 4, 1, 24 / box.diameter(), false};
    raster.render(state.view(), box, camera, 24, 2, false, NeuralVertexStorage::Packed, pixels.p,
                  nullptr);
    auto full = pixels.download();
    raster.render(state.view(), box, camera, 24, 2, false, NeuralVertexStorage::Packed, pixels.p,
                  nullptr, nullptr, true, words.p);
    auto mask = pixels.download();
    auto bits = words.download();
    require(raster.surfaces().mask && !raster.surfaces().attributes,
            "mask memory fixture retained attribute surfaces");
    for (size_t i = 0; i < mask.size(); ++i)
        require(mask[i].covered == full[i].covered &&
                    mask[i].covered == ((bits[i / 32] >> (i % 32)) & 1) && !mask[i].visible,
                "instrumented mask/full coverage differs");
}
// Diagnostic-only stress: retain worker Vulkan sessions across geometry and
// target churn, then make both workers begin normal resource destruction at the
// same barrier. No action topology workspace or full corpus mesh is required.
enum class Retirement { Concurrent, DestroySerial, JoinSerial };
static void teardown_contracts(Retirement mode) {
    NeuralOptions options;
    options.memory_mib = 768;
    options.raster_backend = NeuralRasterBackend::Vulkan;
    options.vertex_storage = NeuralVertexStorage::Packed;
    options.mask_only_coverage = true;
    options.view_batch = 4;
    MemoryBudget shared{size_t(options.memory_mib) << 20, 0, 0, 0};
    std::mutex retirement;
    for (uint32_t round = 0; round < 12; ++round) {
        std::barrier quiesced(2);
        training::WorkerRetirement joined_retirement;
        std::array<bool, 2> parked{};
        auto worker = [&](uint32_t id) {
            struct Arrival {
                std::barrier<>& barrier;
                bool arrived{};
                ~Arrival() {
                    if (!arrived)
                        barrier.arrive_and_drop();
                }
            } arrival{quiesced};
            // A diagnostic control only: when selected, keep the lock through
            // normal native teardown, after both streams have quiesced.
            std::unique_lock retire(retirement, std::defer_lock);
            TeardownEnd resources_end{"fixture.worker.resources", &id};
            gpu::StreamScope stream;
            MemoryScope memory(shared);
            TeardownEnd session_end{"fixture.worker.session", &id};
            AuditSession session(options);
            WeightsData weights;
            weights.architecture = conditioned_placement_schema;
            weights.hidden_width = 64;
            weights.values.resize(policy_weights(weights.architecture, weights.hidden_width));
            TeardownEnd policy_end{"fixture.worker.policy", &id};
            ActionCuda policy(weights, options, 1024);
            {
                for (uint32_t job = 0; job < 3; ++job) {
                    Mesh mesh;
                    const uint32_t width = std::array{16u, 256u, 768u}[(job + id + round) % 3];
                    for (uint32_t y = 0; y < width; ++y)
                        for (uint32_t x = 0; x < width; ++x)
                            mesh.positions.push_back(
                                {float(x) / (width - 1), float(y) / (width - 1), 0});
                    for (uint32_t y = 0; y + 1 < width; ++y)
                        for (uint32_t x = 0; x + 1 < width; ++x) {
                            const auto at = y * width + x;
                            mesh.indices.insert(mesh.indices.end(), {at, at + 1, at + width, at + 1,
                                                                     at + width + 1, at + width});
                        }
                    mesh.double_sided = {1};
                    auto previous = mesh;
                    previous.indices.resize(previous.indices.size() - 6);
                    const auto box = bounds(mesh.view());
                    gpu::Device device(options);
                    auto positions = gpu::upload_stream(device, mesh.view().positions);
                    gpu::Buffer<uint32_t> indices(device, mesh.indices.size());
                    gpu::Buffer<uint8_t> sided(device, mesh.double_sided.size());
                    indices.upload(mesh.indices);
                    sided.upload(mesh.double_sided);
                    DeviceMeshView candidate{};
                    candidate.positions = positions.p;
                    candidate.indices = indices.p;
                    candidate.double_sided = sided.p;
                    candidate.vertices = uint32_t(mesh.positions.size());
                    candidate.faces = uint32_t(mesh.view().triangles());
                    candidate.sided_count = 1;
                    candidate.identity = (1ull << 62) + round * 32 + id * 12 + job * 4;
                    std::array<DeviceMeshView, 4> lanes;
                    for (size_t lane = 0; lane < lanes.size(); ++lane) {
                        lanes[lane] = candidate;
                        lanes[lane].identity += lane;
                    }
                    AuditCuda audit(options, mesh.view());
                    audit.bind_reference(previous.view());
                    EvalSettings settings;
                    settings.profile = Profile::Coverage;
                    settings.views = {4, 0, 817};
                    settings.limit = 3;
                    settings.max_changed_area = .2;
                    settings.supersample = settings.max_supersample = 4;
                    for (double screen : {64., 256., 128., 64.}) {
                        settings.screen_size = screen;
                        audit.with_candidate_rasters(lanes, [&] {
                            const auto source =
                                audit.certify_candidates(mesh.view(), lanes, box, settings);
                            auto adjacent = settings;
                            adjacent.limit = 2;
                            const auto prior =
                                audit.certify_candidates(previous.view(), lanes, box, adjacent);
                            for (size_t lane = 0; lane < lanes.size(); ++lane)
                                require(source[lane].valid && prior[lane].valid &&
                                            !source[lane].value.resource_limited &&
                                            !prior[lane].value.resource_limited &&
                                            !source[lane].value.cancelled &&
                                            !prior[lane].value.cancelled,
                                        "teardown fixture audit incomplete");
                        });
                    }
                }
                gpu::check(cudaStreamSynchronize(stream.value));
            }
            teardown_trace("fixture.worker.quiesced", &id, "begin");
            quiesced.arrive_and_wait();
            teardown_trace("fixture.worker.quiesced", &id, "end");
            arrival.arrived = true;
            if (mode == Retirement::DestroySerial)
                retire.lock();
            if (mode == Retirement::JoinSerial) {
                parked[id] = true;
                joined_retirement.park(uint8_t(id));
            }
            teardown_trace("fixture.worker.resources", &id, "begin");
        };
        if (mode == Retirement::JoinSerial) {
            std::array<std::exception_ptr, 2> failures;
            std::array<std::thread, 2> threads;
            uint32_t created = 0;
            try {
                for (; created < threads.size(); ++created)
                    threads[created] = std::thread([&, id = created] {
                        try {
                            worker(id);
                        } catch (...) {
                            failures[id] = std::current_exception();
                            if (!parked[id])
                                joined_retirement.park(uint8_t(id));
                        }
                    });
            } catch (...) {
                for (auto missing = created; missing < threads.size(); ++missing)
                    quiesced.arrive_and_drop();
                joined_retirement.join(threads);
                throw;
            }
            joined_retirement.join(threads);
            for (auto failure : failures)
                if (failure)
                    std::rethrow_exception(failure);
        } else {
            auto first = std::async(std::launch::async, worker, 0),
                 second = std::async(std::launch::async, worker, 1);
            first.get();
            second.get();
        }
        require(shared.live == 0 && shared.peak > 0 && shared.peak <= shared.limit,
                "teardown fixture leaked or exceeded shared budget");
        std::cout << "teardown round " << round << " passed, peak=" << shared.peak.load()
                  << " bytes\n"
                  << std::flush;
    }
}
int main(int argc, char** argv) {
    try {
        bool memory_boundaries = argc == 2 && std::string_view(argv[1]) == "--memcheck";
        bool teardown = argc == 2 && std::string_view(argv[1]) == "--teardown";
        bool serial_teardown = argc == 2 && std::string_view(argv[1]) == "--teardown-serial";
        bool joined_teardown = argc == 2 && std::string_view(argv[1]) == "--teardown-join";
        if (argc != 1 && !memory_boundaries && !teardown && !serial_teardown && !joined_teardown)
            throw std::invalid_argument(
                "expected --memcheck, --teardown, --teardown-serial or --teardown-join");
        if (!neural_available())
            return 77;
        if (teardown || serial_teardown || joined_teardown) {
            require(allow_debugger_attach(), "explicit debugger attach request was rejected");
            teardown_contracts(joined_teardown   ? Retirement::JoinSerial
                               : serial_teardown ? Retirement::DestroySerial
                                                 : Retirement::Concurrent);
            return 0;
        }
        NeuralOptions options;
        options.memory_mib = 512;
        options.raster_backend = NeuralRasterBackend::Vulkan;
        auto stage = [&](const char* name, auto&& work) {
            if (memory_boundaries)
                std::cout << name << " begin" << std::endl;
            work();
            if (memory_boundaries)
                std::cout << name << " passed" << std::endl;
        };
        stage("draw domains", [&] { draw_domain_contracts(memory_boundaries); });
        stage("packed seeds", packed_seed_domain);
        stage("serial candidates", [&] { candidate_contracts(memory_boundaries); });
        stage("grouped candidate rasters", [&] { grouped_candidate_contracts(memory_boundaries); });
        // Each worker must retain the serial verdicts on an independent stream.
        MemoryBudget shared{size_t(384) << 20, 0, 0, 0};
        std::barrier ready(2);
        auto worker = [&] {
            gpu::StreamScope stream;
            MemoryScope memory(shared);
            ready.arrive_and_wait();
            candidate_contracts(memory_boundaries);
        };
        stage("concurrent candidates", [&] {
            auto first = std::async(std::launch::async, worker),
                 second = std::async(std::launch::async, worker);
            first.get();
            second.get();
        });
        require(shared.live == 0 && shared.peak > 0 && shared.peak <= shared.limit,
                "worker shared budget/lifetime contract");
        if (memory_boundaries) {
            stage("mask buffers", mask_memory_contracts);
            std::cout << "Vulkan mask, candidate boundaries and concurrent ownership memory "
                         "contracts passed\n";
            return 0;
        }
        auto mesh = fixture();
        auto original = mesh;
        auto b = bounds(mesh.view());
        Camera c{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, b.radius * 4, 1, 24 / b.diameter(), false};
        for (auto storage : {NeuralVertexStorage::Float32, NeuralVertexStorage::Position16,
                             NeuralVertexStorage::Packed}) {
            options.vertex_storage = storage;
            auto image = raster_gpu(mesh.view(), b, c, 24, 2, false, options);
            unsigned visible = 0, covered = 0;
            for (auto p : image.pixels) {
                covered += p.covered;
                if (p.visible) {
                    ++visible;
                    require(p.covered && p.material == 65535,
                            "hardware visibility/material contract");
                    require(p.normal.z > .999, "hardware front normal");
                    require(std::abs(p.color.x - 31 / 255.f) < 1e-5, "RGBA8 linear color");
                }
            }
            require(visible > 100 && covered >= visible && !image.clipped,
                    "hardware empty or clipped front face");
            auto back = mesh;
            std::swap(back.indices[1], back.indices[2]);
            std::swap(back.indices[4], back.indices[5]);
            auto culled = raster_gpu(back.view(), b, c, 24, 2, false, options);
            for (auto p : culled.pixels)
                require(!p.covered && !p.visible, "one-sided back face not culled");
            auto two = raster_gpu(back.view(), b, c, 24, 2, true, options);
            unsigned back_count = 0;
            for (auto p : two.pixels)
                if (p.visible) {
                    ++back_count;
                    require(p.normal.z < -.999, "two-sided normal flip");
                }
            require(back_count == visible, "two-sided visibility differs");
            auto flat = mesh;
            flat.normals.clear();
            auto flat_image = raster_gpu(flat.view(), b, c, 24, 2, false, options);
            for (auto p : flat_image.pixels)
                if (p.visible)
                    require(p.normal.z > .999, "missing-normal flat fallback");
            flat.normals.assign(4, {});
            auto zero_image = raster_gpu(flat.view(), b, c, 24, 2, false, options);
            for (auto p : zero_image.pixels)
                if (p.visible)
                    require(p.normal.z > .999, "zero-normal flat fallback");
            auto outside = mesh;
            outside.positions[0].x -= 100;
            require(raster_gpu(outside.view(), b, c, 24, 2, true, options).clipped,
                    "out-of-frame hardware draw escaped audit");
        }
        options.vertex_storage = NeuralVertexStorage::Automatic;
        require(options.draw_storage() == NeuralVertexStorage::Packed,
                "hardware default must pack draw streams");
        auto bad = mesh;
        bad.uv[0].x = std::nextafter(8.f, INFINITY);
        bool rejected = false;
        try {
            (void)raster_gpu(bad.view(), b, c, 24, 2, true, options);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "default packed draw silently clamped out-of-range UV");
        EvalSettings e;
        e.screen_size = 24;
        e.supersample = 2;
        e.max_supersample = 4;
        e.limit = 3;
        e.views = {2, 1, 51};
        e.profile = Profile::Attributes;
        auto identical = evaluate_gpu(mesh.view(), mesh.view(), b, e, options);
        require(identical.complete && identical.passed && identical.views_evaluated == 3,
                "packed identity bypassed raster audit");
        // Same hardware images: threshold certificates must agree with the exact
        // metric, including limits immediately around the measured boundary.
        NeuralStats stats;
        AuditCuda audit(options, mesh.view());
        for (auto profile : {Profile::Coverage, Profile::Normals, Profile::Attributes})
            for (float displacement : {0.f, .03f, .4f}) {
                auto changed = mesh;
                changed.positions[0].x += displacement;
                changed.normals[0] = normalized({displacement, 0, 1});
                GpuActionState state(changed.view(), options, true);
                e.profile = profile;
                for (double limit : {.8, 1.5, 3.}) {
                    e.limit = limit;
                    auto exact = audit.evaluate(mesh.view(), state.view(), b, e);
                    auto predicate = audit.certify(mesh.view(), state.view(), b, e, &stats);
                    require(predicate.verdict !=
                                (exact.passed ? AuditVerdict::Fail : AuditVerdict::Pass),
                            "sparse certificate contradicts exact hardware metric");
                    require(predicate.verdict != AuditVerdict::Unknown,
                            "bounded fixture returned unknown");
                }
            }
        require(stats.gpu_sparse_passes > 0 && stats.gpu_sparse_queries > 0,
                "sparse tests did not exercise witnesses");
        // Match exact metrics and certificate verdicts across independent switches.
        // Limits below the sparse certificate floor exercise exact fallback/refinement.
        for (auto profile : {Profile::Coverage, Profile::Normals, Profile::Attributes})
            for (float shift : {0.f, .13f, .6f})
                for (double limit : {.4, 1.5, 3.}) {
                    auto changed = mesh;
                    changed.positions[0].x += shift;
                    changed.normals[0] = normalized({shift, 0, 1});
                    e.profile = profile;
                    e.limit = limit;
                    auto serial = options;
                    serial.view_batch = 1;
                    serial.direct_targets = false;
                    serial.mask_only_coverage = false;
                    AuditCuda control(serial, mesh.view());
                    GpuActionState state(changed.view(), serial, true);
                    auto expected = control.evaluate(mesh.view(), state.view(), b, e);
                    auto certificate = control.certify(mesh.view(), state.view(), b, e);
                    for (uint8_t batch : {2, 4})
                        for (bool direct : {false, true}) {
                            auto settings = options;
                            settings.view_batch = batch;
                            settings.direct_targets = direct;
                            AuditCuda tested(settings, mesh.view());
                            auto actual = tested.evaluate(mesh.view(), state.view(), b, e);
                            auto predicate = tested.certify(mesh.view(), state.view(), b, e);
                            require(actual.passed == expected.passed &&
                                        actual.complete == expected.complete &&
                                        actual.error == expected.error &&
                                        actual.coverage == expected.coverage &&
                                        actual.changed_area == expected.changed_area &&
                                        actual.normal_degrees == expected.normal_degrees &&
                                        actual.views_evaluated == expected.views_evaluated &&
                                        actual.supersample == expected.supersample,
                                    "batched exact audit differs from serial");
                            require(predicate.verdict == certificate.verdict &&
                                        predicate.changed_area == certificate.changed_area &&
                                        predicate.views == certificate.views &&
                                        predicate.supersample == certificate.supersample,
                                    "direct/batched certificate differs from serial");
                        }
                    bool pruned = false;
                    auto expected_pruning =
                        control.certify(mesh.view(), state.view(), b, e, nullptr, .01, &pruned);
                    AuditCuda batched(options, mesh.view());
                    bool batch_pruned = false;
                    auto actual_pruning = batched.certify(mesh.view(), state.view(), b, e, nullptr,
                                                          .01, &batch_pruned);
                    require(pruned == batch_pruned &&
                                expected_pruning.verdict == actual_pruning.verdict &&
                                expected_pruning.views == actual_pruning.views,
                            "batched incumbent pruning differs from serial");
                }
        {
            auto precise = mesh;
            precise.positions[0].x = -.81234567f;
            precise.positions[2].y = .91234567f;
            precise.exact_position_bits = {15};
            auto controls = options;
            controls.exact_position_bps = 10000;
            GpuActionState exact_state(precise.view(), controls, true);
            gpu::Device device(controls);
            VulkanRaster renderer(controls);
            gpu::Buffer<AuditPixel> pixels(device, 64 * 64);
            auto view = exact_state.view();
            renderer.render(view, b, c, 24, 2, false, NeuralVertexStorage::Packed, pixels.p,
                            nullptr);
            auto packed = pixels.download();
            renderer.render(view, b, c, 24, 2, false, NeuralVertexStorage::Float32, pixels.p,
                            nullptr);
            auto full = pixels.download();
            for (size_t i = 0; i < full.size(); ++i)
                require(packed[i].covered == full[i].covered &&
                            packed[i].visible == full[i].visible &&
                            packed[i].normal.x == full[i].normal.x &&
                            packed[i].normal.y == full[i].normal.y &&
                            packed[i].normal.z == full[i].normal.z,
                        "indexed sparse precision lookup changed exact positions");
            auto layout = draw_layout(view, NeuralVertexStorage::Packed);
            gpu::Buffer<char> draw(device, layout.bytes);
            pack_draw(view, NeuralVertexStorage::Packed, layout, draw.p);
            auto bytes = draw.download();
            for (size_t i = 0; i < precise.positions.size(); ++i)
                require(std::memcmp(bytes.data() + layout.exact_positions + i * sizeof(Vec3),
                                    &precise.positions[i], sizeof(Vec3)) == 0,
                        "sparse exact positions were rounded or reordered");
            view.exact_position_bps = 500;
            bool rejected = false;
            try {
                renderer.render(view, b, c, 24, 2, false, NeuralVertexStorage::Packed, pixels.p,
                                nullptr);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "renderer did not enforce precision cap");
        }
        e.cancelled = [] { return true; };
        GpuActionState state(mesh.view(), options, true);
        require(audit.certify(mesh.view(), state.view(), b, e).verdict == AuditVerdict::Unknown,
                "cancelled query produced label");
        require(same_mesh_data(mesh.view(), original.view()), "packing mutated source");
        {
            MemoryScope memory(options);
            gpu::Device device(options);
            GpuActionState state(mesh.view(), options, true);
            auto view = state.view();
            auto layout = draw_layout(view, NeuralVertexStorage::Packed);
            gpu::Buffer<char> draw(device, layout.bytes);
            pack_draw(view, NeuralVertexStorage::Packed, layout, draw.p);
            auto bytes = draw.download();
            auto read = [&]<class T>(size_t at) {
                T v;
                std::memcpy(&v, bytes.data() + at, sizeof(v));
                return v;
            };
            require(layout.position_stride == 6 && layout.normal_stride == 4,
                    "packed stream stride changed");
            for (size_t i = 0; i < 4; ++i) {
                auto p = mesh.positions[i];
                require(read.template operator()<uint16_t>(layout.position + 6 * i) ==
                            (p.x < 0 ? 0 : 65535),
                        "GPU position code differs at bound");
                require(read.template operator()<uint16_t>(layout.uv + 4 * i) ==
                            (mesh.uv[i].x < 0 ? 0 : 65535),
                        "GPU UV endpoint code");
                require(read.template operator()<uint32_t>(layout.normal + 4 * i) == (511u << 20),
                        "GPU RGB10 normal layout");
                require(read.template operator()<uint32_t>(layout.tangent + 4 * i) ==
                            (511u | (3u << 30)),
                        "GPU tangent alpha sign layout");
                auto color = read.template operator()<ColorRGBA8>(layout.color + 4 * i);
                require(color.r == 31 && color.a == 47, "GPU RGBA8 byte layout");
            }
            // Reuse the same attachments across removal, depth ties, material seams
            // and bounds changes. Changed revisions must repack and redraw all faces.
            Mesh layers;
            layers.positions = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0},
                                {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
            layers.normals.assign(6, {0, 0, 1});
            layers.indices = {0, 1, 2, 3, 4, 5};
            layers.materials = {7, 13};
            auto positions = gpu::upload_stream(device, layers.view().positions);
            auto normals = gpu::upload_stream(device, layers.view().normals);
            gpu::Buffer<uint32_t> indices(device, layers.indices.size());
            indices.upload(layers.indices);
            gpu::Buffer<uint16_t> materials(device, layers.materials.size());
            materials.upload(layers.materials);
            DeviceMeshView dm{};
            dm.positions = positions.p;
            dm.normals = normals.p;
            dm.indices = indices.p;
            dm.materials = materials.p;
            dm.vertices = 6;
            dm.faces = 2;
            dm.identity = 0xfeed;
            dm.revision = 1;
            VulkanRaster raster(options);
            gpu::Buffer<AuditPixel> pixels(device, 64 * 64);
            auto center = [&] {
                raster.render(dm, b, c, 24, 2, false, NeuralVertexStorage::Packed, pixels.p,
                              nullptr);
                return pixels.download()[32 * 64 + 32];
            };
            require(center().material == 7, "hardware depth tie does not keep first face");
            auto views = cameras(b, 24, {2, 2, 171});
            std::array<gpu::Buffer<AuditPixel>, 4> batched;
            std::array<gpu::Buffer<RasterDebugPixel>, 4> debug;
            std::array<RasterOutput, 4> outputs;
            for (size_t i = 0; i < 4; ++i) {
                batched[i] = gpu::Buffer<AuditPixel>(device, 64 * 64);
                debug[i] = gpu::Buffer<RasterDebugPixel>(device, 64 * 64);
                outputs[i] = {batched[i].p, nullptr, debug[i].p};
            }
            raster.render_batch(dm, b, views, 24, 2, true, NeuralVertexStorage::Packed, outputs);
            std::array<std::vector<AuditPixel>, 4> expected;
            std::array<std::vector<RasterDebugPixel>, 4> witnesses;
            for (size_t i = 0; i < 4; ++i) {
                expected[i] = batched[i].download();
                witnesses[i] = debug[i].download();
            }
            for (size_t i = 0; i < 4; ++i) {
                bool clipped =
                    raster.render(dm, b, views[i], 24, 2, true, NeuralVertexStorage::Packed,
                                  pixels.p, nullptr, debug[0].p);
                auto actual = pixels.download();
                auto actual_debug = debug[0].download();
                require(clipped == outputs[i].clipped, "batch clip flag differs");
                for (size_t j = 0; j < actual.size(); ++j) {
                    auto x = actual[j], y = expected[i][j];
                    require(x.covered == y.covered && x.visible == y.visible &&
                                x.material == y.material && x.normal.x == y.normal.x &&
                                x.normal.y == y.normal.y && x.normal.z == y.normal.z,
                            "batched draw changes pixels");
                    require(actual_debug[j].face == witnesses[i][j].face &&
                                actual_debug[j].depth == witnesses[i][j].depth,
                            "debug attachment changes depth/ownership");
                    if (x.visible)
                        require(actual_debug[j].face < 2 && actual_debug[j].depth >= 0 &&
                                    actual_debug[j].depth <= 1,
                                "debug face/depth range");
                }
            }
            gpu::Buffer<uint32_t> mask_bits(device, 64 * 64 / 32);
            raster.trim_targets(1);
            auto full_bytes = raster.bytes();
            raster.render(dm, b, c, 24, 2, false, NeuralVertexStorage::Packed, pixels.p, nullptr,
                          nullptr, true, mask_bits.p);
            auto mask_image = pixels.download();
            auto words = mask_bits.download();
            auto mask_bytes = raster.bytes();
            require(mask_bytes < full_bytes, "mask-only retained shading/depth targets");
            for (size_t i = 0; i < mask_image.size(); ++i)
                require(mask_image[i].covered == ((words[i / 32] >> (i % 32)) & 1) &&
                            !mask_image[i].visible,
                        "bit mask or absent visibility contract");
            auto mask_surface = raster.surfaces();
            require(mask_surface.mask && !mask_surface.attributes && !mask_surface.colors,
                    "mask-only exported shading surfaces");
            center();
            auto full_image = pixels.download();
            for (size_t i = 0; i < mask_image.size(); ++i)
                require(mask_image[i].covered == full_image[i].covered,
                        "mask-only changed coverage");
            raster.trim_targets(1);
            require(center().material == 7, "target reclamation invalidated surviving slot");
            // Domain changes are part of the geometry cache key even without a revision.
            dm.fixed_quantization = true;
            dm.quant_low = {-2, -2, 0};
            dm.quant_extent = {4, 4, 0};
            require(center().material == 7, "explicit quantization domain failed");
            dm.quant_extent.z = -1;
            bool invalid_domain = false;
            try {
                center();
            } catch (const std::invalid_argument&) {
                invalid_domain = true;
            }
            require(invalid_domain, "invalid quantization domain reused cached geometry");
            dm.fixed_quantization = false;
            std::vector<uint32_t> removed = {3, 4, 5, 0, 1, 2};
            indices.upload(removed);
            std::vector<uint16_t> new_materials = {13, 7};
            materials.upload(new_materials);
            dm.faces = 1;
            ++dm.revision;
            require(center().material == 13, "removed foreground left stale depth or material");
            for (unsigned i = 3; i < 6; ++i)
                layers.positions[i].x += 10;
            positions.upload(layers.positions);
            ++dm.revision;
            require(!center().visible, "changed bounds reused stale packed geometry");
        }
        std::cout << "Vulkan draw/storage contracts passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
