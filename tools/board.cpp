// Build a portable visual board from recorded outputs. This is a presentation
// exporter, not a renderer or a substitute for the benchmark's visual gates.
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;

static std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
static json read_json(const fs::path& path) { return json::parse(read_text(path)); }
static std::string sha256(std::span<const uint8_t> bytes) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> hash{};
    SHA256(bytes.data(), bytes.size(), hash.data());
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (auto c : hash) { out += hex[c >> 4]; out += hex[c & 15]; }
    return out;
}
static std::string base64(std::span<const uint8_t> bytes) {
    if (bytes.size() > 512u * 1024u * 1024u) throw std::runtime_error("Board buffer too large");
    std::string out(4 * ((bytes.size() + 2) / 3) + 1, '\0');
    auto count = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(out.data()),
                                bytes.data(), static_cast<int>(bytes.size()));
    out.resize(static_cast<size_t>(count));
    return out;
}
static uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
static void append_u32(Bytes& out, uint32_t n) {
    for (unsigned shift = 0; shift != 32; shift += 8) out.push_back(uint8_t(n >> shift));
}
// Accept the uncompressed float/u32 accessors emitted by our own glTF exporter.
static Bytes accessor(const json& gltf, std::span<const uint8_t> binary,
                      size_t index, unsigned component, const char* type, size_t element) {
    const auto& a = gltf.at("accessors").at(index);
    if (a.at("componentType") != component || a.at("type") != type || a.contains("sparse"))
        throw std::runtime_error("Unexpected board accessor format");
    const auto& view = gltf.at("bufferViews").at(a.at("bufferView").get<size_t>());
    const size_t count = a.at("count"), length = view.at("byteLength");
    const size_t offset = a.value("byteOffset", size_t(0));
    const size_t start = view.value("byteOffset", size_t(0));
    const size_t stride = view.value("byteStride", element);
    if (view.at("buffer") != 0 || stride < element || start > binary.size() ||
        length > binary.size() - start || offset > length ||
        (count && (element > length - offset || count - 1 > (length - offset - element) / stride)))
        throw std::runtime_error("Board accessor exceeds buffer");
    Bytes out;
    out.reserve(count * element);
    for (size_t i = 0; i != count; ++i) {
        auto p = binary.begin() + static_cast<ptrdiff_t>(start + offset + i * stride);
        out.insert(out.end(), p, p + static_cast<ptrdiff_t>(element));
    }
    return out;
}
struct Geometry {
    Bytes positions, indices;
    std::array<float, 3> low{
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    std::array<float, 3> high{
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
};
static Geometry geometry(const json& gltf, std::span<const uint8_t> binary, size_t lod) {
    Geometry out;
    for (const auto& primitive : gltf.at("meshes").at(lod).at("primitives")) {
        if (primitive.value("mode", 4) != 4) throw std::runtime_error("Board requires triangles");
        auto p = accessor(gltf, binary, primitive.at("attributes").at("POSITION"), 5126, "VEC3", 12);
        auto ix = accessor(gltf, binary, primitive.at("indices"), 5125, "SCALAR", 4);
        if (ix.size() % 12) throw std::runtime_error("Incomplete board triangle");
        const size_t count = p.size() / 12, base = out.positions.size() / 12;
        if (base + count > UINT32_MAX) throw std::runtime_error("Board vertex range overflow");
        for (size_t i = 0; i < ix.size(); i += 4) {
            uint32_t index = u32(ix.data() + i);
            if (index >= count) throw std::runtime_error("Invalid board vertex index");
            append_u32(out.indices, static_cast<uint32_t>(base + index));
        }
        for (size_t i = 0; i < p.size(); i += 4) {
            float f = std::bit_cast<float>(u32(p.data() + i));
            if (!std::isfinite(f)) throw std::runtime_error("Nonfinite board position");
            size_t axis = (i / 4) % 3;
            out.low[axis] = std::min(out.low[axis], f);
            out.high[axis] = std::max(out.high[axis], f);
        }
        out.positions.insert(out.positions.end(), p.begin(), p.end());
    }
    return out;
}
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4) throw std::invalid_argument("blitz-board REPOSITORY_ROOT [CONFIG OUTPUT_DIRECTORY]");
        const fs::path root = fs::absolute(argv[1]);
        auto config = read_json(argc==4?fs::path(argv[2]):root / "research/board/examples.json");
        const fs::path output_dir=argc==4?fs::path(argv[3]):root/"research/board";
        fs::create_directories(output_dir);
        auto corpus = read_json(root / "research/corpus.json");
        auto scan_rights = read_json(root / "research/scan-rights.json").at("items");
        const auto run = config.at("run").get<std::string>();
        const auto run_dir = root / "research/runs" / run;
        auto meta = read_json(run_dir / "metadata.json");
        json board{{"date", config.at("date")}, {"run", run}, {"settings", meta.at("config")},
                   {"run_sha256", meta.at("run_sha256")}, {"assets", json::array()},
                   {"corpus_count", corpus.at("assets").size()}};
        json manifest{{"version", 1}, {"run", run}, {"run_sha256", meta.at("run_sha256")},
                      {"rendering", "Untextured geometric face normals; presentation views are not audit cameras"},
                      {"assets", json::array()}};
        const auto collection=config.contains("foliage")?read_json(root/config.at("foliage").get<std::string>()):json();
        const auto collection_text=collection.dump();const auto collection_hash=sha256({reinterpret_cast<const uint8_t*>(collection_text.data()),collection_text.size()});
        unsigned foliage_chains=0;
        for (auto example : config.at("examples")) {
            const auto id = example.at("id").get<std::string>();
            const auto example_run=example.value("run",run);const auto example_dir=root/"research/runs"/example_run;
            const auto example_meta=example_run==run?meta:read_json(example_dir/"metadata.json");
            auto row = read_json(example_dir / "rows" / (id + ".json"));
            if (!row.at("complete").get<bool>() || row.value("failed", false) ||
                row.at("run_sha256") != example_meta.at("run_sha256"))
                throw std::runtime_error("Incomplete or mismatched example row: " + id);
            auto gltf_text=read_text(example_dir / "meshes" / id / "chain.gltf");auto gltf=json::parse(gltf_text);
            auto storage = read_text(example_dir / "meshes" / id / "chain.bin");
            std::span<const uint8_t> binary(reinterpret_cast<const uint8_t*>(storage.data()), storage.size());
            auto source = std::find_if(corpus.at("assets").begin(), corpus.at("assets").end(),
                                      [&](const auto& a) { return a.at("id") == id; });
            if (source == corpus.at("assets").end()) {
                if(collection.is_null()||example_meta.value("scope",std::string())!="foliage_card_geometry_only"||example_meta.at("source_manifest_sha256")!=collection_hash||row.at("output_sha256")!=sha256(binary)||row.at("gltf_sha256")!=sha256({reinterpret_cast<const uint8_t*>(gltf_text.data()),gltf_text.size()}))throw std::runtime_error("Missing foliage chain provenance: "+id);
                auto asset=std::find_if(collection.at("assets").begin(),collection.at("assets").end(),[&](const auto& a){return a.at("id")==id;});
                if(asset==collection.at("assets").end()||asset->at("geometry").at("triangles")!=row.at("result").at("lods").at(0).at("triangles"))throw std::runtime_error("Missing foliage source");
                auto package=std::find_if(collection.at("packages").begin(),collection.at("packages").end(),[&](const auto& p){return p.at("id")==asset->at("package");});
                if(package==collection.at("packages").end())throw std::runtime_error("Missing foliage package");
                example["source_url"]=package->at("source_url");example["license"]=package->at("license");example["license_url"]=package->at("license_url");example["audit_scope"]="card_geometry_only";++foliage_chains;
            }else{
                example["source_url"] = source->at("source_url");
                const auto identity = source->at("source_identity").get<std::string>();
                if (scan_rights.contains(identity)) example["source_url"] = scan_rights.at(identity).at("url");
                example["license"] = source->at("license");example["license_url"] = source->at("license_url");example["audit_scope"]="archived_opaque_v1";
            }
            example["run"]=example_run;
            example["ratio"] = row.at("ratio");
            example["output_mode"] = row.contains("output_mode")?row.at("output_mode"):example_meta.at("config").at("output");
            example["bake_seconds"] = row.value("generation_seconds",row.at("seconds").get<double>());
            example["bake_timing_scope"] = row.contains("generation_seconds") ? "generation_and_audit" : "import_generation_audit_export";
            example["lods"] = json::array();
            auto rows = row.at("result").at("lods");
            if (gltf.at("nodes").size() != rows.size()) throw std::runtime_error("LOD node count mismatch");
            example["runtime_levels"]=row.at("result").value("runtime_levels",json::array());
            json record{{"id", id}, {"chain_bin_sha256", sha256(binary)},
                        {"source_url", example.at("source_url")}, {"license", example.at("license")},
                        {"license_url", example.at("license_url")}, {"credit", example.at("credit")},
                        {"output_sha256", row.at("output_sha256")}, {"triangles", json::array()},
                        {"output_mode",example.at("output_mode")},{"bake_seconds",example.at("bake_seconds")},
                        {"bake_timing_scope",example.at("bake_timing_scope")},{"run",example_run},{"run_sha256",example_meta.at("run_sha256")},{"audit_scope",example.at("audit_scope")}};
            size_t previous = SIZE_MAX;
            for (size_t i = 0; i < rows.size(); ++i) {
                auto g = geometry(gltf, binary, gltf.at("nodes").at(i).at("mesh").get<size_t>());
                size_t triangles = g.indices.size() / 12;
                if (triangles != rows[i].at("triangles") || triangles > previous ||
                    !rows[i].at("source").at("passed").get<bool>() ||
                    !rows[i].at("adjacent").at("passed").get<bool>())
                    throw std::runtime_error("Geometry or acceptance mismatch: " + id);
                previous = triangles;
                record["triangles"].push_back(triangles);
                if (!i) {
                    example["low"] = g.low;
                    example["high"] = g.high;
                    // Match the evaluator's source frame: rounded AABB center,
                    // then the furthest source position, not the box diagonal.
                    std::array<float, 3> center{};
                    for (size_t axis = 0; axis != 3; ++axis)
                        center[axis] = float((double(g.low[axis]) + g.high[axis]) * .5);
                    double radius_sq = 0;
                    for (size_t p = 0; p < g.positions.size(); p += 12) {
                        double distance_sq = 0;
                        for (size_t axis = 0; axis != 3; ++axis) {
                            double d = double(std::bit_cast<float>(u32(g.positions.data() + p + axis * 4))) - center[axis];
                            distance_sq += d * d;
                        }
                        radius_sq = std::max(radius_sq, distance_sq);
                    }
                    example["center"] = center;
                    example["radius"] = std::sqrt(radius_sq);
                }
                auto level = rows[i];
                level["positions"] = base64(g.positions);
                level["indices"] = base64(g.indices);
                example["lods"].push_back(std::move(level));
            }
            board["assets"].push_back(std::move(example));
            manifest["assets"].push_back(std::move(record));
        }
        board["foliage_chains"]=foliage_chains;
        if(config.contains("vegetation"))board["vegetation"]=read_json(root/config.at("vegetation").get<std::string>());
        board["scores"] = json::object();
        for (const auto* name : {"round1-qem", "baseline-meshopt", "baseline-fastquadric",
                                "baseline-cgal-probabilistic", "round2-coupled",
                                "round2-hybrid", "profile-normals", "corpus-smoke"}) {
            auto summary = read_json(root / "research/runs" / name / "summary.json");
            if (!summary.at("complete").get<bool>()) throw std::runtime_error("Incomplete score");
            board["scores"][name] = summary;
        }
        board["microbench"] = read_json(root / "research/microbench-optimized.json");
        if(config.contains("round4")) {
            board["round4"]=read_json(root/config.at("round4").get<std::string>());
            for(auto& item:board["round4"].at("comparisons")) {
                std::string name=item.at("run");
                auto summary=read_json(root/"research/runs"/name/"summary.json");
                if(!summary.at("complete").get<bool>())throw std::runtime_error("Incomplete comparison");
                board["scores"][name]=summary;
            }
        }
        if(config.contains("foliage")) {
            auto file=root/config.at("foliage").get<std::string>();
            auto checks=read_json(file.parent_path()/"previews/checks.json");auto canonical=collection.dump();
            const auto hash=sha256({reinterpret_cast<const uint8_t*>(canonical.data()),canonical.size()});
            if(checks.at("manifest_sha256")!=hash||collection.at("benchmark_eligible")!=false||!checks.at("errors").empty())throw std::runtime_error("Unverified foliage previews");
            json gallery={{"assets",json::array()},{"targets",collection.at("targets")},{"manifest_sha256",hash}};
            for(auto& asset:collection.at("assets")) {
                const std::string id=asset.at("id");if(id.find_first_of("/\\.")!=id.npos)throw std::runtime_error("Invalid foliage id");
                auto package=std::find_if(collection.at("packages").begin(),collection.at("packages").end(),[&](const auto& p){return p.at("id")==asset.at("package");});
                auto checked=std::find_if(checks.at("assets").begin(),checks.at("assets").end(),[&](const auto& a){return a.at("id")==id;});
                auto png=read_text(file.parent_path()/"previews"/(id+".png"));std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(png.data()),png.size());
                if(package==collection.at("packages").end()||checked==checks.at("assets").end()||checked->at("sha256")!=sha256(bytes)||checked->at("triangles")!=asset.at("geometry").at("triangles"))throw std::runtime_error("Foliage preview provenance mismatch");
                gallery["assets"].push_back({{"id",id},{"name",asset.at("name")},{"category",asset.at("category")},{"triangles",asset.at("geometry").at("triangles")},
                    {"image","data:image/png;base64,"+base64(bytes)},{"source_url",package->at("source_url")},{"license_url",package->at("license_url")},{"license",package->at("license")},
                    {"requires_material_setup",asset.at("requires_material_setup")},{"material_setup",asset.at("material_setup")},{"preview_note",asset.at("preview").at("mask_note")},
                    {"source_group",asset.at("source_group")},{"model",asset.at("model")},{"selector",asset.at("selector")},{"bake_seconds",nullptr},{"bake_status","not_baked"},{"vertex_mode","original_source"}});
            }
            auto catalog=read_text(root/"tools/foliage.html"),serialized=gallery.dump();
            for(size_t p=0;(p=serialized.find('<',p))!=std::string::npos;p+=6)serialized.replace(p,1,"\\u003c");
            auto marker=catalog.find("@FOLIAGE_DATA@");if(marker==catalog.npos)throw std::runtime_error("Missing catalog data token");catalog.replace(marker,14,serialized);
            std::ofstream catalog_file(output_dir/"catalog.html");catalog_file<<catalog;if(!catalog_file)throw std::runtime_error("Cannot write foliage catalog");
            board["foliage_source_count"]=collection.at("assets").size();
            manifest["foliage"]={{"manifest_sha256",hash},{"chain_count",foliage_chains},{"audit_scope","Opaque card geometry only; opacity and shading are not audited; no SCORE"}};
        }
        auto text = read_text(root / "tools/board.html");
        auto data = board.dump();
        // Protect the script element even if a future caption contains HTML.
        for (size_t p = 0; (p = data.find('<', p)) != std::string::npos; p += 6) data.replace(p, 1, "\\u003c");
        constexpr auto token = "@BOARD_DATA@";
        auto position = text.find(token);
        if (position == std::string::npos) throw std::runtime_error("Missing board data token");
        text.replace(position, std::char_traits<char>::length(token), data);
        auto output = output_dir / "index.html";
        std::ofstream html(output, std::ios::binary);
        html << text;
        if (!html) throw std::runtime_error("Cannot write board");
        std::ofstream manifest_file(output_dir / "manifest.json");
        manifest_file << manifest.dump(2) << '\n';
        if (!manifest_file) throw std::runtime_error("Cannot write board manifest");
        std::cout << output << " (" << text.size() << " bytes; " << board["assets"].size()
                  << " examples, " << meta.at("config").at("levels") << " levels each)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
