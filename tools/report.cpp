#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using json=nlohmann::json;
namespace fs=std::filesystem;
static json read(const fs::path& p){std::ifstream f(p);json j;f>>j;return j;}
static std::string escape(std::string s){std::string out;for(char c:s){if(c=='&')out+="&amp;";else if(c=='<')out+="&lt;";else if(c=='>')out+="&gt;";else if(c=='"')out+="&quot;";else out+=c;}return out;}
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::invalid_argument("blitz-report EXPERIMENTS.json OUTPUT_DIRECTORY");
        auto input=read(argv[1]);fs::path output=argv[2];fs::create_directories(output);
        std::ofstream md(output/"RESULTS.md"),svg(output/"scores.svg");
        md<<"# Recorded experiments\n\n"<<input.at("description").get<std::string>()<<"\n\n"
          <<"SCORE = 100 × (1 − category-balanced mean retained triangle ratio). Compare rows only within the stated scenario. Failed assets remain in the denominator. An incomplete run has no score.\n\n"
          <<"| Round / variant | Scenario | Complete | SCORE | Seconds | Fallbacks | Failed assets |\n|---|---|---:|---:|---:|---:|---:|\n";
        auto& runs=input.at("runs");size_t height=100+52*runs.size();
        svg<<"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\""<<height<<"\" viewBox=\"0 0 1000 "<<height<<"\"><rect width=\"100%\" height=\"100%\" fill=\"#111827\"/><g font-family=\"sans-serif\" fill=\"#e5e7eb\"><text x=\"24\" y=\"32\" font-size=\"21\">BlitzRemesher — audited research scenarios</text><text x=\"24\" y=\"56\" font-size=\"13\">Higher score means fewer triangles within that row's fixed visual gates. See RESULTS.md for scenarios.</text>";
        unsigned index=0;
        for(auto& run:runs) {
            std::string label=run.at("label");fs::path dir=run.at("path").get<std::string>();double y=90+52*index++;
            svg<<"<text x=\"24\" y=\""<<y+17<<"\" font-size=\"13\">"<<escape(label)<<"</text>";
            if(!fs::exists(dir/"summary.json")){md<<"| "<<label<<" | pending | no | — | — | — | — |\n";continue;}
            auto s=read(dir/"summary.json"),meta=read(dir/"metadata.json");auto c=meta.at("config");
            auto research=c.value("research",json::object());auto placement=research.value("output",json(nullptr));
            std::string output_mode=c.value("output",placement.is_null()?std::string("auto"):placement.get<std::string>());
            std::string scenario=output_mode+"/"+c.at("profile").get<std::string>()+"/"+c.value("chain",research.value("chain",std::string("hybrid")))+", N="+std::to_string(c.at("levels").get<int>())+", budget="+std::to_string(c.at("candidate_budget").get<int>());
            if(output_mode=="auto")scenario+=", overhead bps="+std::to_string(c.at("triangle_overhead_bps").get<int>());
            size_t failed=0;for(auto& row:fs::directory_iterator(dir/"rows"))if(row.path().extension()==".json")failed+=read(row.path()).value("failed",false);
            md<<"| ["<<label<<"]("<<fs::relative(dir,output).generic_string()<<"/summary.json) | "<<scenario<<" | "<<s.at("completed")<<"/"<<s.at("expected")<<" | ";
            if(s["score"].is_null())md<<"—";else md<<std::fixed<<std::setprecision(2)<<s["score"].get<double>();
            md<<" | "<<std::setprecision(2)<<s["seconds"].get<double>()<<" | "<<s.at("fallbacks")<<" | "<<failed<<" |\n";
            if(!s["score"].is_null()){double score=s["score"];svg<<"<rect x=\"390\" y=\""<<y<<"\" width=\""<<score*5<<"\" height=\"25\" rx=\"3\" fill=\"#38bdf8\"/><text x=\""<<400+score*5<<"\" y=\""<<y+18<<"\" font-size=\"14\">"<<std::fixed<<std::setprecision(2)<<score<<"</text>";}
            else svg<<"<text x=\"390\" y=\""<<y+18<<"\" font-size=\"14\">Incomplete — no score</text>";
        }
        svg<<"</g></svg>\n";
        md<<"\nTimings are measured on a shared workstation and include the evaluator and export. External adapters additionally include process startup and PLY interchange. They are not isolated kernel timings. Each run retains raw rows, configuration, camera, input and binary hashes.\n\n![Score chart](scores.svg)\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
