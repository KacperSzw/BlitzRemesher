#pragma once
// Included after the acquisition primitives in corpus.cpp. Expansion writes a
// new manifest, preserving the original family splits and release holdout.
inline std::string neural_family(const std::string& identity){
    return "polyhaven:"+std::regex_replace(std::regex_replace(identity,std::regex("[0-9]+"),""),std::regex("_+$| +$"),"");
}
inline int prepare_neural_corpus_cache(int argc,char** argv){
    if(argc!=3)throw std::invalid_argument("blitz-corpus neural-cache MANIFEST CACHE_DIRECTORY");
    const auto manifest=json::parse(read(argv[1]));const fs::path cache=argv[2];fs::create_directories(cache);
    json report={{"manifest_sha256",hash(read(argv[1]))},{"complete",false},{"assets",json::array()}};size_t failures=0;
    for(const auto& asset:manifest.at("assets")){
        json row={{"id",asset.at("id")}};
        try{
            for(const auto& file:asset.at("files"))if(hash(read(file.at("path").get<std::string>()))!=file.at("sha256").get<std::string>())throw std::invalid_argument("source checksum changed");
            auto prepared=blitz::neural::training::prepared_mesh(asset,cache);row["complete"]=true;row["reference_bytes"]=prepared.provenance.at("reference_bytes");row["packed_bytes"]=prepared.provenance.at("packed_bytes");
        }catch(const std::exception& error){row["complete"]=false;row["error"]=error.what();++failures;}
        report["assets"].push_back(row);std::cout<<row.dump()<<std::endl;
    }
    report["complete"]=failures==0;write(cache/(fs::path(argv[1]).stem().string()+"-preparation.json"),report.dump(2));return failures?2:0;
}
inline json neural_geometry(json a){
    using namespace blitz;auto m=load_mesh(a.at("path").get<std::string>());
    if(m.view().triangles()>2000000||m.view().triangles()<3000)throw std::invalid_argument("new training geometry must have 3000..2000000 triangles");
    // Validate the actual packed representation before counting an acquisition.
    std::ostringstream packed(std::ios::binary);neural::training::write_packed_mesh(packed,m.view());
    auto bytes=[](auto& v){return std::string(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));};
    a["geometry_sha256"]=hash(hash(bytes(m.positions))+hash(bytes(m.indices)));a["vertices"]=m.positions.size();a["triangles"]=m.view().triangles();a["diameter"]=bounds(m.view()).diameter();
    a["normals"]=!m.normals.empty();a["uv"]=!m.uv.empty();a["colors"]=!m.colors.empty();a["import_status"]="ok";a["source_group"]=neural_family(a.at("source_identity"));a.erase("source_metadata");return a;
}
inline int expand_neural_corpus(int argc,char** argv){
    if(argc!=5)throw std::invalid_argument("blitz-corpus neural-expand DATA_ROOT FROZEN_CORPUS TRAIN_SELECTION NEW_DIRECTORY");
    const fs::path root=argv[1],original=argv[2],selection=argv[3],directory=argv[4];
    const auto source=json::parse(read(original)),allowed=json::parse(read(selection));
    const json contract={{"version",2},{"seed",0xB1172026u},{"source_sha256",hash(read(original))},{"selection_sha256",hash(read(selection))},{"training_groups",100},{"validation_groups",20},{"new_triangle_range",{3000,2000000}},{"packed_uv_range",{-8,8}}};
    fs::create_directories(directory);
    if(fs::exists(directory/"contract.json")&&json::parse(read(directory/"contract.json"))!=contract)throw std::invalid_argument("expanded corpus contract changed");
    write(directory/"contract.json",contract.dump(2));
    if(fs::exists(directory/"complete.json")){std::cout<<"Expanded corpus already frozen\n";return 0;}
    json acquired=fs::exists(directory/"acquisition.json")?json::parse(read(directory/"acquisition.json")):json{{"assets",json::array()},{"rejected",json::array()}};
    std::set<std::string> selected,blocked,geometry,attempted;for(const auto& a:allowed.at("assets"))selected.insert(a.at("id"));
    std::map<std::string,json> train,validation;
    for(const auto& a:source.at("assets")){
        auto group=a.at("source_group").get<std::string>();blocked.insert(group);geometry.insert(a.at("geometry_sha256"));
        auto* target=selected.contains(a.at("id"))?&train:a.at("split")=="validation"?&validation:nullptr;
        if(target&&(!target->contains(group)||a.at("triangles").get<uint64_t>()>target->at(group).at("triangles").get<uint64_t>()))(*target)[group]=a;
    }
    for(const auto& a:acquired.at("assets")){auto g=a.at("source_group").get<std::string>();blocked.insert(g);geometry.insert(a.at("geometry_sha256"));(a.at("split")=="validation"?validation:train)[g]=a;}
    for(const auto& a:acquired.at("rejected"))attempted.insert(a.at("id"));
    uint64_t bytes=0;for(const auto& a:source.at("assets"))bytes+=a.at("bytes").get<uint64_t>();for(const auto& a:acquired.at("assets"))bytes+=a.at("bytes").get<uint64_t>();
    curl_global_init(CURL_GLOBAL_DEFAULT);auto catalog=cached("https://api.polyhaven.com/assets?t=models",root/"catalog-polyhaven.json");
    std::map<std::string,std::vector<std::string>> candidates;
    for(auto i=catalog.begin();i!=catalog.end();++i){auto id=i.key();if(blocked.contains(neural_family(id))||attempted.contains("ph_"+id))continue;auto c=i.value().at("categories");
        auto category=contains(c,"rocks")?"rocks":contains(c,"nature")||contains(c,"food")?"organic":"manufactured";candidates[category].push_back(id);}
    for(auto& [category,ids]:candidates)std::sort(ids.begin(),ids.end(),[](const auto& a,const auto& b){return hash("B1172026"+a)<hash("B1172026"+b);});
    const auto began=std::chrono::steady_clock::now();std::map<std::string,size_t> cursor;
    bool progress=true;
    while((train.size()<100||validation.size()<20)&&progress&&std::chrono::steady_clock::now()-began<std::chrono::minutes(50)){
        progress=false;
        for(auto category:{"rocks","organic","manufactured"}){
            auto& ids=candidates[category];auto& at=cursor[category];while(at<ids.size()&&blocked.contains(neural_family(ids[at])))++at;
            if(at==ids.size()||(train.size()>=100&&validation.size()>=20))continue;
            if(bytes+max_file>50ull*1024*1024*1024)throw std::length_error("source data acquisition budget exhausted");
            progress=true;auto id=ids[at++];
            try{auto a=neural_geometry(poly(id,category,root));auto g=a.at("source_group").get<std::string>();
                if(geometry.contains(a.at("geometry_sha256")))throw std::invalid_argument("duplicate geometry");
                a["split"]=validation.size()<20?"validation":"development";bytes+=a.at("bytes").get<uint64_t>();geometry.insert(a.at("geometry_sha256"));blocked.insert(g);
                (a.at("split")=="validation"?validation:train)[g]=a;acquired["assets"].push_back(a);
                std::cout<<"groups: "<<train.size()<<" training, "<<validation.size()<<" validation; "<<id<<std::endl;
            }catch(const std::exception& error){acquired["rejected"].push_back({{"id","ph_"+id},{"reason",error.what()}});std::cerr<<id<<": "<<error.what()<<'\n';}
            write(directory/"acquisition.json",acquired.dump(2));
        }
    }
    curl_global_cleanup();
    if(train.size()!=100||validation.size()!=20){std::cerr<<"Expansion incomplete: "<<train.size()<<" training, "<<validation.size()<<" validation groups\n";return 2;}
    json corpus=source;corpus["version"]=2;corpus["parent_sha256"]=contract.at("source_sha256");for(const auto& a:acquired.at("assets"))corpus["assets"].push_back(a);
    json train_manifest={{"version",2},{"source_groups",100},{"score_eligible",false},{"assets",json::array()}},val_manifest={{"version",2},{"source_groups",20},{"score_eligible",false},{"assets",json::array()}};
    for(const auto& [group,a]:train)train_manifest["assets"].push_back(a);for(const auto& [group,a]:validation)val_manifest["assets"].push_back(a);
    write(directory/"corpus.json",corpus.dump(2));write(directory/"training.json",train_manifest.dump(2));write(directory/"validation.json",val_manifest.dump(2));
    write(directory/"complete.json",json({{"complete",true},{"training_groups",train.size()},{"validation_groups",validation.size()},{"source_bytes",bytes},{"corpus_sha256",hash(read(directory/"corpus.json"))},{"training_sha256",hash(read(directory/"training.json"))},{"validation_sha256",hash(read(directory/"validation.json"))}}).dump(2));return 0;
}
