#pragma once
#include "neural_action_data.hpp"
#include "neural_action_gpu.hpp"
#include "neural_memory.hpp"
#include "neural_cuda.cuh"
#include "neural_json.hpp"
#include "neural_packing.hpp"
#include "neural_training_cache.hpp"
#include "neural_timeline.hpp"
#include <iostream>
namespace blitz::neural::training {
struct PlacementRequest {
    uint32_t states{8},pool{4},previous_steps{},seed{101};
    double pixels{128},minutes{5},source_limit{3},adjacent_limit{3};
    bool sparse{true};std::string policy_sha256;std::function<bool()> cancelled;bool compact_data{true};uint32_t repair_budget{256};
    fs::path corpus{"research/corpus.json"},selection{"research/neural/training-manifest.json"},mesh_cache;Profile profile{Profile::Attributes};
    TrainingCache* cache{};fs::path episode;double retained{1};bool simplifier{};
};
struct PlacementResult {bool complete;ActionData data;json index;CompactActions compact;};
inline PlacementResult prepare_placements(const std::string& asset,const fs::path& output,const PlacementRequest& request,const NeuralOptions& options,ActionCuda* policy=nullptr){
    const auto& [states,pool,previous_steps,seed,pixels,minutes,source_limit,adjacent_limit,sparse,policy_hash,cancelled,compact_data,repair_budget,corpus,selection,mesh_cache,profile,cache,episode,retained,simplifier]=request;
    if(!states||states>4096||previous_steps>4096||!pool||pool>16||!std::isfinite(pixels)||pixels<16||pixels>512||!std::isfinite(minutes)||minutes<=0||minutes>50||!std::isfinite(source_limit)||source_limit<=0||source_limit>16||!std::isfinite(adjacent_limit)||adjacent_limit<=0||adjacent_limit>16)throw std::invalid_argument("preparation bounds");
    if(fs::exists(output/"contract.json"))throw std::invalid_argument("choose a fresh placement dataset directory");fs::create_directories(output);
    auto load_start=std::chrono::steady_clock::now();double proposals_seconds=0,trial_seconds=0,audit_seconds=0,commit_seconds=0;
    auto timed=[](double& total,auto&& fn){auto t=std::chrono::steady_clock::now();auto value=fn();total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return value;};
    MemoryScope memory(options);auto metadata=cache?cache->metadata(asset):training_metadata(asset,corpus,selection);Mesh mesh,prepared;json cache_provenance;std::shared_ptr<const PreparedMesh> borrowed;
    if(cache){borrowed=cache->get(asset);cache_provenance=borrowed->provenance;}else if(!mesh_cache.empty()&&options.draw_storage()==NeuralVertexStorage::Packed){auto cached=prepared_mesh(metadata,mesh_cache);mesh=std::move(cached.source);prepared=std::move(cached.draw);cache_provenance=std::move(cached.provenance);}else mesh=load_mesh(metadata.at("path").get<std::string>());
    auto source=borrowed?borrowed->source.view():mesh.view();auto prepared_view=borrowed?borrowed->draw.view():prepared.view();auto bounds=blitz::bounds(source);AuditSession session(options);AuditCuda audit(options,source);NeuralStats stats;ActionData data;data.architecture=placement_schema;
    double load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-load_start).count();auto start=std::chrono::steady_clock::now();auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto cancel=[&]{return (request.cancelled&&request.cancelled())||seconds()>minutes*60;};
    auto e=action_eval(previous_steps?std::min(512.,pixels*2):pixels,source_limit,profile);e.cancelled=cancel;auto adjacent=e;adjacent.limit=adjacent_limit;Lod previous;MeshView previous_view=source;bool emitted=false,exhausted=false,unknown=false;uint64_t queries=0,reused_queries=0,reused_adjacent=0,pruned_queries=0,proven_losers=0;uint32_t accepted=0;
    json contract={{"training_profile",training_profile_name(profile)},{"mask_only_coverage",options.mask_only_coverage},{"candidate_batch",options.candidate_batch},{"data_storage",compact_data?"compact-v1":"fp32"},{"repair_budget",repair_budget},{"quantization","fixed mesh bounds; grid placements"},{"schema",placement_schema},{"teacher_version",sparse?4:3},{"raster",raster_name(options.raster_backend)},{"vertex_storage",storage_name(options.draw_storage())},{"metric_kind",sparse?"threshold bounds; exact committed states":"exact"},{"seed",seed},{"asset",metadata},{"source_manifest_sha256",cache?cache->corpus_hash:file_sha256(corpus)},{"training_selection_sha256",cache?cache->selection_hash:file_sha256(selection)},{"protocol_sha256",cache?cache->protocol_hash:file_sha256("research/PROTOCOL.md")},{"binary_sha256",cache?cache->binary_hash:file_sha256("/proc/self/exe")},{"states_requested",states},{"pool",pool},{"pixels",pixels},{"previous_steps",previous_steps},{"source_limit",source_limit},{"adjacent_limit",adjacent_limit},{"area_limit",e.max_changed_area},{"views",{6,2}},{"view_seed",e.views.rotation_seed},{"supersample",4},{"max_supersample",8},{"gpu_memory_mib",options.memory_mib},{"teacher_candidates",(profile==Profile::Coverage?10:20)+(policy?1:0)},{"policy_payload_sha256",request.policy_sha256},{"prepared_mesh",cache_provenance},{"target","one audited joint XYZ/normal candidate per edge; no averaging"},{"preference",sparse?"minimum triangles, then coverage-area bound, then stable proposal order":"minimum triangles, then minimum worst normalized source/adjacent error"}};
    contract["episode"]={{"kind",simplifier?"audited_simplifier":episode.empty()?"source":"audited_policy_trajectory"},{"parent_sha256",episode.empty()?"":file_sha256(episode)},{"target_retained",retained}};
    write_json(output/"contract.json",contract);json trace=json::array();
    auto packing_start=std::chrono::steady_clock::now();auto destination=e;destination.screen_size=pixels;auto baseline=repair_packing(source,options,e,repair_budget,prepared_view,previous_steps?&destination:nullptr);write_json(output/"packing.json",baseline.diagnostics.is_null()?json{{"initial",measurement_json(baseline.initial)},{"final",measurement_json(baseline.final)},{"trials",0}}:baseline.diagnostics);
    if(!baseline.final.passed||!baseline.final.complete){save_actions(output/"actions.bin",data,compact_data);json index={{"schema",placement_schema},{"complete",false},{"status",baseline.final.resource_limited?"resource_failure":cancel()?"cancelled":"representation_failure"},{"asset",asset},{"category",metadata.at("category")},{"contract_sha256",file_sha256(output/"contract.json")},{"path","actions.bin"},{"sha256",file_sha256(output/"actions.bin")},{"states",0},{"queries",0},{"accepted",0},{"source_triangles",source.triangles()},{"teacher_triangles",source.triangles()},{"reference_confirmed",false},{"seconds",seconds()},{"training_started",false},{"baseline",measurement_json(baseline.final)}};write_json(output/"index.json",index);write_json(output/"trajectory.json",trace);return {false,std::move(data),index};}
    // Seeds are proposals, never targets. Every seed passes the unchanged
    // original-source audit; a rejected seed remains visible and uses LOD0.
    if(!std::isfinite(retained)||retained<=0||retained>1)throw std::invalid_argument("episode retained fraction");
    auto quantization=vertex_bounds(source);std::unique_ptr<GpuActionState> prepared_state;
    json seed_result={{"requested",contract.at("episode")},{"accepted",false}};
    if(!episode.empty()||simplifier){Mesh candidate;
        if(!episode.empty()){std::ifstream f(episode,std::ios::binary);candidate=read_packed_mesh(f).first;}
        else {ReduceSettings r;r.target_triangles=std::max<size_t>(1,size_t(source.triangles()*retained));r.output=OutputMode::Rebuild;r.coupled_wedges=true;r.cancelled=cancel;auto reduced=reduce(baseline.mesh.view(),r);candidate=copy_mesh(reduced.view(baseline.mesh.view()));}
        seed_result["triangles"]=candidate.view().triangles();
        // Audit the exact packed working representation. A QEM proposal may
        // extend beyond the source bounds; the UNorm grid maps it into the
        // supported domain before evaluation. Unsupported UVs/precision remain
        // visible seed rejections, never fatal errors or accepted labels.
        try{auto proposed=std::make_unique<GpuActionState>(candidate.view(),options,true,&quantization,source);
            auto measured=audit.evaluate(source,proposed->view(),bounds,e,&stats);auto d=previous_steps?audit.evaluate(source,proposed->view(),bounds,destination,&stats):measured;
            seed_result["audit"]=measurement_json(measured);seed_result["destination"]=measurement_json(d);
            if(measured.complete&&measured.passed&&d.complete&&d.passed&&candidate.view().triangles()<=source.triangles()){prepared_state=std::move(proposed);baseline.mesh=std::move(candidate);seed_result["accepted"]=true;}
        }catch(const std::invalid_argument& error){seed_result["rejection"]=error.what();}
    }
    seed_result["start_retained"]=double(baseline.mesh.view().triangles())/source.triangles();write_json(output/"seed.json",seed_result);
    double packing_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-packing_start).count();auto state_start=std::chrono::steady_clock::now();if(!prepared_state)prepared_state=std::make_unique<GpuActionState>(baseline.mesh.view(),options,true,&quantization,source);auto& state=*prepared_state;double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-state_start).count();
    struct TeacherAudit {bool known{},passed{};double error{},changed_area{};};
    auto query=[&](MeshView reference,DeviceMeshView candidate,const EvalSettings& config,double cutoff,bool* pruned){
        if(sparse){auto p=timed(audit_seconds,[&]{return audit.certify(reference,candidate,bounds,config,&stats,cutoff,pruned);});return TeacherAudit{p.verdict!=AuditVerdict::Unknown,p.verdict==AuditVerdict::Pass,p.error_upper,p.changed_area};}
        auto m=timed(audit_seconds,[&]{return audit.evaluate(reference,candidate,bounds,config,&stats,cutoff,pruned);});return TeacherAudit{action_audit_known(m,config),m.passed,m.error,m.changed_area};
    };
    for(uint32_t step=0;step<states+previous_steps&&!cancel();++step){
        if(previous_steps&&step==previous_steps){previous=state.snapshot();auto a=audit.evaluate(source,previous.data.view(),bounds,e,&stats),d=audit.evaluate(source,previous.data.view(),bounds,destination,&stats);if(!a.complete||!d.complete){unknown=true;break;}if(!a.passed||!d.passed||previous.data.view().triangles()>=source.triangles()){exhausted=true;break;}previous_view=previous.data.view();emitted=true;e.screen_size=pixels;adjacent.screen_size=pixels;}
        double fraction=std::array{.1,.5,.9,.02}[step%4];auto rows=timed(proposals_seconds,[&]{return state.teacher_actions(condition(e,adjacent.limit,fraction),pool,seed,policy);});if(rows.empty()){exhausted=true;break;}
        size_t first=data.labels.size(),chosen=SIZE_MAX;uint32_t best_faces=UINT32_MAX;double best_margin=INFINITY;std::vector<GpuActionState::Proposal> winners;std::vector<Action> actions;
        json state_trace={{"revision",step},{"triangles",state.view().faces},{"previous_triangles",previous_view.triangles()},{"pixels",e.screen_size},{"queries",json::array()}};
        for(auto& row:rows){bool found=false;uint8_t best_label=0;double best=INFINITY;uint32_t faces=UINT32_MAX;GpuActionState::Proposal winner{};
            auto alternatives=timed(proposals_seconds,[&]{return state.teacher_proposals(row.action,profile!=Profile::Coverage);});
            struct Query {TeacherAudit source,adjacent;uint32_t faces{};bool valid{},pruned{};};std::array<Query,21> evaluated{};
            std::array<bool,21> ready{};
            auto batch_candidates=[&](size_t first){
                size_t count=std::min<size_t>(options.candidate_batch,alternatives.size()-first);
                for(;;){std::vector<GpuActionState::Proposal> proposals;std::vector<size_t> ids;
                    for(size_t i=first;i<first+count;++i){size_t prior=0;while(prior<i&&std::memcmp(&alternatives[i].placement,&alternatives[prior].placement,sizeof(Placement)))++prior;if(prior==i){ids.push_back(i);proposals.push_back(alternatives[i]);}}
                    if(proposals.empty())return;
                    try{auto views=timed(trial_seconds,[&]{return state.trial_batch(row.action,proposals);});double cutoff=found&&(best_label&27)==27?best:INFINITY;
                        auto a=timed(audit_seconds,[&]{Timeline range("candidate-audit");return audit.certify_candidates(source,views,bounds,e,&stats,cutoff);});std::vector<CandidateAudit> b=a;
                        if(emitted||e.limit!=adjacent.limit){std::vector<DeviceMeshView> active;std::vector<size_t> lanes;for(size_t i=0;i<views.size();++i)if(a[i].valid&&!a[i].pruned){active.push_back(views[i]);lanes.push_back(i);}if(!active.empty()){auto values=timed(audit_seconds,[&]{return audit.certify_candidates(previous_view,active,bounds,adjacent,&stats,cutoff);});for(size_t i=0;i<lanes.size();++i)b[lanes[i]]=values[i];}}
                        else reused_adjacent+=std::count_if(a.begin(),a.end(),[](auto& x){return x.valid&&!x.pruned;});
                        auto value=[](const CandidateAudit& x){return TeacherAudit{x.value.verdict!=AuditVerdict::Unknown,x.value.verdict==AuditVerdict::Pass,x.value.error_upper,x.value.changed_area};};
                        for(size_t i=0;i<ids.size();++i){evaluated[ids[i]]={value(a[i]),value(b[i]),a[i].faces,a[i].valid,a[i].pruned||b[i].pruned};ready[ids[i]]=true;}
                        ++stats.candidate_batches;stats.candidate_batch_proposals+=proposals.size();return;
                    }catch(const gpu::ResourceError&){if(count<=1)return;count=(count+1)/2;}
                }
            };
            static_assert(sizeof(Placement)==9*sizeof(float));
            for(size_t index=0;index<alternatives.size()&&!cancel();++index){
                // The topology/triangle count is identical for every placement
                // of this edge. Coverage-area bounds are nonnegative, and ties
                // keep the first proposal. A certified zero is unbeatable.
                if(sparse&&found&&(best_label&27)==27&&best==0){proven_losers+=alternatives.size()-index;break;}
                auto& proposal=alternatives[index];auto& q=evaluated[index];size_t prior=0;
                while(prior<index&&std::memcmp(&proposal.placement,&alternatives[prior].placement,sizeof(Placement)))++prior;
                if(prior<index){q=evaluated[prior];++reused_queries;}
                else {if(!ready[index]&&index&&sparse&&options.raster_backend==NeuralRasterBackend::Vulkan&&options.candidate_batch>1)batch_candidates(index);
                    if(ready[index]){double cutoff=found&&(best_label&27)==27?best:INFINITY;if(std::isfinite(cutoff)&&((e.max_changed_area>0&&q.source.changed_area/e.max_changed_area>=cutoff)||(adjacent.max_changed_area>0&&q.adjacent.changed_area/adjacent.max_changed_area>=cutoff)))q.pruned=true;}
                    else {DeviceMeshView candidate;q.valid=timed(trial_seconds,[&]{return state.trial(row.action,proposal.placement,candidate);});if(q.valid){q.faces=candidate.faces;
                    double cutoff=found&&(best_label&27)==27?best:INFINITY;
                    q.source=query(source,candidate,e,cutoff,std::isfinite(cutoff)?&q.pruned:nullptr);
                    if(!q.pruned){if(!emitted&&e.limit==adjacent.limit){q.adjacent=q.source;++reused_adjacent;}else q.adjacent=query(previous_view,candidate,adjacent,cutoff,std::isfinite(cutoff)?&q.pruned:nullptr);}}}}
                if(!q.valid)continue;auto& a=q.source;auto& b=q.adjacent;++queries;
                if(q.pruned){++pruned_queries;state_trace["queries"].push_back({{"from",row.action.from},{"to",row.action.to},{"candidate",index},{"known_mask",0},{"pruned_by_incumbent",true},{"faces",q.faces}});continue;}
                bool ka=a.known,kb=b.known;uint8_t label=uint8_t((ka?SourceKnown:0)|(kb?AdjacentKnown:0)|(ka&&a.passed?PlacementSourcePass:0)|(kb&&b.passed?PlacementAdjacentPass:0));bool safe=(label&27)==27;
                double margin=std::max({a.error/e.limit,b.error/adjacent.limit,a.changed_area/e.max_changed_area,b.changed_area/adjacent.max_changed_area});if(sparse&&safe)margin=std::max(a.changed_area/e.max_changed_area,b.changed_area/adjacent.max_changed_area);if(!std::isfinite(margin))margin=INFINITY;
                state_trace["queries"].push_back({{"from",row.action.from},{"to",row.action.to},{"candidate",index},{"known_mask",label},{"source_error",a.error},{"adjacent_error",b.error},{"faces",q.faces}});
                if(!found||(safe&&((best_label&3)!=3))||(safe==((best_label&3)==3)&&margin<best)){found=true;winner=proposal;best_label=label;best=margin;faces=q.faces;}
                if(!ka||!kb){unknown=true;break;}
            }
            if(found){bool safe=(best_label&27)==27;if(safe)best_label|=uint8_t(PositionKnown|(winner.normal_mask&1?Normal0Known:0)|(winner.normal_mask&2?Normal1Known:0));
                data.x.insert(data.x.end(),row.x.begin(),row.x.end());data.labels.push_back(best_label);data.from.push_back(row.action.from);data.to.push_back(row.action.to);
                for(unsigned j=0;j<9;++j)data.targets.push_back((best_label&(32u<<(j/3)))?winner.target[j]:0);
                actions.push_back(row.action);winners.push_back(winner);if(safe&&(faces<best_faces||(faces==best_faces&&best<best_margin))){chosen=actions.size()-1;best_faces=faces;best_margin=best;}}
            if(unknown||cancel())break;
        }
        if(data.labels.size()>first){data.offsets.push_back(uint32_t(data.labels.size()));data.progress.push_back(float(1-double(state.view().faces)/source.triangles()));}
        if(unknown||cancel()){state_trace["complete"]=false;trace.push_back(state_trace);break;}
        if(chosen==SIZE_MAX){exhausted=true;state_trace["stopped"]="no feasible queried placement";trace.push_back(state_trace);break;}
        if(sparse){DeviceMeshView candidate;if(!state.trial(actions[chosen],winners[chosen].placement,candidate))throw std::runtime_error("teacher finalist became invalid");
            auto source_check=audit.evaluate(source,candidate,bounds,e,&stats),adjacent_check=!emitted&&e.limit==adjacent.limit?source_check:audit.evaluate(previous_view,candidate,bounds,adjacent,&stats);
            if(!source_check.complete||!source_check.passed||!adjacent_check.complete||!adjacent_check.passed){unknown=true;state_trace["complete"]=false;state_trace["exact_confirmation_failed"]=true;trace.push_back(state_trace);break;}}
        // Every accepted predecessor must also remain valid when the camera
        // scale changes. A visual rejection is an audited terminal outcome.
        if(previous_steps&&!emitted){DeviceMeshView candidate;if(!state.trial(actions[chosen],winners[chosen].placement,candidate))throw std::runtime_error("predecessor finalist became invalid");auto d=audit.evaluate(source,candidate,bounds,destination,&stats);if(!d.complete){unknown=true;break;}if(!d.passed){exhausted=true;state_trace["stopped"]="destination_screen_gate";trace.push_back(state_trace);break;}}
        data.labels[first+chosen]|=PlacementPreferred;timed(commit_seconds,[&]{state.commit(actions[chosen],winners[chosen].placement);return true;});++accepted;state_trace["selected"]={{"from",actions[chosen].from},{"to",actions[chosen].to},{"triangles",state.view().faces}};trace.push_back(state_trace);
        std::cout<<json({{"state",step},{"queries",queries},{"triangles",state.view().faces},{"seconds",seconds()}}).dump()<<std::endl;
    }
    auto final=state.view();auto a=audit.evaluate(source,final,bounds,e,&stats),b=!emitted&&e.limit==adjacent.limit?a:audit.evaluate(previous_view,final,bounds,adjacent,&stats);
    bool complete=!cancel()&&!unknown&&(data.states()==states+previous_steps||exhausted)&&a.complete&&a.passed&&b.complete&&b.passed;
    CompactActions encoded;if(compact_data){encoded=compact_actions(data);std::ofstream f(output/"actions.bin",std::ios::binary);write_compact(f,encoded);f.close();if(!f)throw std::runtime_error("compact teacher output failed");}else save_actions(output/"actions.bin",data,false);
    if(complete){auto end=state.snapshot();std::ofstream f(output/"episode.bin",std::ios::binary);write_packed_mesh(f,end.view(source),&quantization);f.close();if(!f)throw std::runtime_error("episode output failed");}
    write_json(output/"trajectory.json",trace);write_json(output/"reuse.json",{{"duplicate_proposals",reused_queries},{"identical_adjacent_audits",reused_adjacent},{"pruned_candidates",pruned_queries},{"unbeatable_incumbent_skips",proven_losers}});
    write_json(output/"index.json",{{"schema",placement_schema},{"status",complete?(previous_steps&&!emitted?"predecessor_unavailable":exhausted?"search_exhausted":"complete"):cancel()?"cancelled":(a.resource_limited||b.resource_limited||stats.resource_failures)?"resource_failure":unknown?"unknown_audit":"final_audit_failed"},{"baseline",measurement_json(baseline.final)},{"source_audit",measurement_json(a)},{"adjacent_audit",measurement_json(b)},{"complete",complete},{"asset",asset},{"category",metadata.at("category")},{"contract_sha256",file_sha256(output/"contract.json")},{"path","actions.bin"},{"sha256",file_sha256(output/"actions.bin")},{"states",data.states()},{"queries",queries},{"accepted",accepted},{"source_triangles",source.triangles()},{"teacher_triangles",final.faces},{"reference_confirmed",a.complete&&a.passed&&b.complete&&b.passed},{"requested_condition_available",!previous_steps||emitted},{"preceding_lod_emitted",emitted},{"previous_triangles",previous_view.triangles()},{"seconds",seconds()},{"training_started",false},{"timings",{{"load_and_session_seconds",load_seconds},{"packing_and_baseline_seconds",packing_seconds},{"state_setup_seconds",setup_seconds},{"features_and_proposals_seconds",proposals_seconds},{"candidate_build_seconds",trial_seconds},{"candidate_audit_seconds",audit_seconds},{"commit_seconds",commit_seconds},{"other_preparation_seconds",std::max(0.,seconds()-packing_seconds-setup_seconds-proposals_seconds-trial_seconds-audit_seconds-commit_seconds)}}},{"audit",neural_json(stats)}});
    auto index=read_json(output/"index.json");index["seed"]=seed_result;if(complete)index["episode_sha256"]=file_sha256(output/"episode.bin");write_json(output/"index.json",index);
    return {complete,std::move(data),std::move(index),std::move(encoded)};
}
}
