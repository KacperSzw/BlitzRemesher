// Compile in the pinned shell: c++ -std=c++20 -O2 research/precision/summarize.cpp -o build/precision-summary
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
static json read(const fs::path& path){std::ifstream f(path);json j;f>>j;return j;}
static json distribution(std::vector<double> values) {
    auto raw=values;std::sort(values.begin(),values.end());
    return {{"samples",raw},{"median",values[values.size()/2]},{"minimum",values.front()},{"maximum",values.back()}};
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("precision-summary RUN_DIRECTORY");
        json report={{"runs",json::object()},{"full_packed_agreement",json::array()},{"variant_comparisons",json::array()},
            {"timing_repetitions",json::object()},{"failures",json::array()}};
        std::map<std::string,json> rows,metadata;
        std::regex name_pattern("([a-z]+)-(.+)-([0-9]+)");
        for(auto& entry:fs::directory_iterator(argv[1])) {
            if(!entry.is_directory()||!fs::exists(entry.path()/"metadata.json"))continue;
            auto name=entry.path().filename().string();std::smatch match;if(!std::regex_match(name,match,name_pattern))continue;
            auto meta=read(entry.path()/"metadata.json");metadata[name]=meta;
            auto source_hash=meta.at("build").at("source_tree_sha256");
            if(report.contains("source_tree_sha256")&&report["source_tree_sha256"]!=source_hash)throw std::runtime_error("source tree changed: "+name);
            report["source_tree_sha256"]=source_hash;
            json stats={{"variant",match[1].str()},{"scenario",match[2].str()},{"repeat",std::stoi(match[3].str())},
                {"complete",false},{"score",nullptr},{"failed",0},{"fallbacks",0},{"generation_seconds",0.},{"wall_seconds",0.},{"peak_rss_kib",0},
                {"stage_seconds",{{"reduction",0.},{"raster",0.},{"distance",0.}}},{"numerics",json::object()},
                {"max_source_error_px",0.},{"max_adjacent_error_px",0.},{"max_changed_area",0.},{"verified_gates",0}};
            std::map<std::string,std::array<double,3>> tails;
            rows[name]=json::object();
            for(auto& item:fs::directory_iterator(entry.path()/"rows"))if(item.path().extension()==".json") {
                auto row=read(item.path());rows[name][row.at("id").get<std::string>()]=row;
                stats["generation_seconds"]=stats["generation_seconds"].get<double>()+row.value("generation_seconds",0.);
                stats["wall_seconds"]=stats["wall_seconds"].get<double>()+row.value("seconds",0.);
                stats["peak_rss_kib"]=std::max(stats["peak_rss_kib"].get<uint64_t>(),row.value("peak_rss_kib",uint64_t(0)));
                stats["failed"]=stats["failed"].get<unsigned>()+row.value("failed",false);
                stats["fallbacks"]=stats["fallbacks"].get<unsigned>()+row.value("fallback",false);
                if(row.value("failed",false))report["failures"].push_back({{"run",name},{"asset",row["id"]},{"failure",row["failure"]}});
                if(row.contains("stage_seconds"))for(auto& [key,value]:row["stage_seconds"].items())stats["stage_seconds"][key]=stats["stage_seconds"][key].get<double>()+value.get<double>();
                if(row.contains("numerics"))for(auto& [key,value]:row["numerics"].items())stats["numerics"][key]=stats["numerics"].value(key,uint64_t(0))+value.get<uint64_t>();
                auto& tail=tails[row.at("category").get<std::string>()];tail[0]+=row.value("final_ratio",1.);tail[1]+=row.value("last_three_ratio",1.);tail[2]+=1;
                if(!row.value("failed",false)&&row.contains("result"))for(auto& lod:row["result"]["lods"])for(auto gate:{"source","adjacent"}) {
                    auto& check=lod[gate];
                    if(!check.value("passed",false)||!check.value("complete",false)||check.value("nonfinite_error",true)
                       ||check.at("error_px").get<double>()>lod.at(std::string(gate)=="source"?"source_limit":"transition_limit").get<double>())
                        throw std::runtime_error("delivered LOD failed visual gate: "+name);
                    stats["verified_gates"]=stats["verified_gates"].get<unsigned>()+1;
                    auto key=std::string("max_")+gate+"_error_px";
                    stats[key]=std::max(stats[key].get<double>(),check.value("error_px",0.));
                    stats["max_changed_area"]=std::max(stats["max_changed_area"].get<double>(),check.value("changed_area",0.));
                }
            }
            if(fs::exists(entry.path()/"summary.json")) {
                auto summary=read(entry.path()/"summary.json");stats["complete"]=summary["complete"];stats["score"]=summary["score"];
                if(stats["complete"]==true&&summary["expected"]!=rows[name].size())throw std::runtime_error("complete run has missing rows: "+name);
            }
            stats["final_ratio"]=nullptr;stats["last_three_ratio"]=nullptr;
            if(stats["complete"]==true) {
                double final=0,last=0;for(auto& [category,tail]:tails){final+=tail[0]/tail[2];last+=tail[1]/tail[2];}
                stats["final_ratio"]=final/tails.size();stats["last_three_ratio"]=last/tails.size();
            }
            stats["assets"]=rows[name].size();report["runs"][name]=stats;
        }
        for(auto& [name,asset_rows]:rows) {
            auto& stats=report["runs"][name];auto variant=stats["variant"].get<std::string>();
            if(variant=="packed"||stats["scenario"].get<std::string>().starts_with("baseline-"))continue;
            auto reference="packed-"+stats["scenario"].get<std::string>()+"-"+std::to_string(stats["repeat"].get<unsigned>());
            if(!rows.contains(reference))reference="packed-"+stats["scenario"].get<std::string>()+"-1";
            if(!rows.contains(reference))continue;
            for(auto key:{"manifest_sha256","config_sha256","camera_sha256","protocol_sha256","input_format","split","limit"})
                if(metadata[name][key]!=metadata[reference][key])throw std::runtime_error("incomparable scenario: "+name+" "+key);
            unsigned matched=0,changed=0;bool identical=true;
            for(auto& [id,row]:asset_rows.items())if(rows[reference].contains(id)) {
                auto& ref=rows[reference][id];++matched;
                if(row.value("canonical_attributes_sha256",std::string{})!=ref.value("canonical_attributes_sha256",std::string{}))throw std::runtime_error("canonical input changed: "+name+" "+id);
                bool same=row.value("output_sha256",std::string{})==ref.value("output_sha256",std::string{})&&row.value("attributes_sha256",std::string{})==ref.value("attributes_sha256",std::string{});
                changed+=!same;identical&=same&&row.value("result",json{})==ref.value("result",json{});
            }
            json comparison={{"run",name},{"reference",reference},{"matched_assets",matched},{"changed_outputs",changed},{"identical_results",identical}};
            report[variant=="full"?"full_packed_agreement":"variant_comparisons"].push_back(comparison);
            if(variant=="full"&&!identical)throw std::runtime_error("packed/full disagreement: "+name);
        }
        std::map<std::string,std::map<std::string,std::vector<double>>> timings;
        std::map<std::string,std::vector<unsigned>> repeats;
        for(auto& [name,stats]:report["runs"].items()) {
            unsigned repeat=stats["repeat"];if(repeat<2||repeat>4||stats["complete"]!=true)continue;
            auto group=stats["variant"].get<std::string>()+"-"+stats["scenario"].get<std::string>();
            auto reference=group+"-1";
            if(!rows.contains(reference))reference=group+"-2";
            for(auto& [id,row]:rows[name].items())if(rows[reference].contains(id)) {
                auto& ref=rows[reference][id];
                for(auto key:{"canonical_attributes_sha256","output_sha256","attributes_sha256","result"})
                    if(row.at(key)!=ref.at(key))throw std::runtime_error("repeated output changed: "+name+" "+id);
            }
            repeats[group].push_back(repeat);
            for(auto key:{"generation_seconds","wall_seconds","peak_rss_kib"})timings[group][key].push_back(stats[key]);
            for(auto& [key,value]:stats["stage_seconds"].items())timings[group][key+"_seconds"].push_back(value);
        }
        for(auto& [group,metrics]:timings) {
            report["timing_repetitions"][group]["repeats"]=repeats[group];
            report["timing_repetitions"][group]["complete"]=repeats[group]==std::vector<unsigned>{2,3,4};
            for(auto& [metric,values]:metrics)report["timing_repetitions"][group][metric]=distribution(values);
        }
        std::cout<<report.dump(2)<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
