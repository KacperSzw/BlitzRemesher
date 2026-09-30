#pragma once
#include "neural_resident.hpp"
#include "neural_checkpoint.hpp"
#include "neural_cycle_time.hpp"
#include "neural_cycle_history.hpp"
namespace blitz::neural::training {
inline void resident_contracts(bool compact,UpdateBackend backend=UpdateBackend::Reference){
    auto check=[](bool x,const char* why){if(!x)throw std::runtime_error(why);};torch::manual_seed(519);torch::Device device(torch::kCUDA);
    auto page=[](unsigned states,unsigned rows){ActionData d;d.architecture=placement_schema;d.offsets={0};for(unsigned s=0;s<states;++s){for(unsigned r=0;r<rows;++r){
            auto start=d.x.size();d.x.resize(start+placement_features);for(unsigned c=0;c<placement_features;++c){auto slot=feature_slot(c);d.x[start+c]=slot>=0?float((c+r*3+s)%17)/32:slot>=-32?float((c+r)%2):float(c+s)/100;}
            d.x[start+78]=d.x[start+23];d.x[start+79]=d.x[start+47];d.labels.push_back(r==0?255:r==1?8:r==2?16:0);for(unsigned t=0;t<9;++t)d.targets.push_back((d.labels.back()&(32u<<(t/3)))?.125f:0);d.from.push_back(r);d.to.push_back(r+1);}
        d.offsets.push_back(uint32_t(d.labels.size()));d.progress.push_back(float(s%4)/4);}return d;};
    ResidentDataset data(device,281,compact);auto a=page(3,2);data.append(a,"a","first");ActionNetwork model(placement_schema);model->to(device);ResidentUpdate update(model,data,31,{},backend);update.capture();check(update.pool()==2&&update.captures==1,"resident pool includes padding");
    auto initial=update.optimizer.snapshot();update.run(3);check(update.optimizer.state().step==3,"resident graph did not update");auto checkpoint=update.optimizer.snapshot();update.run(4);auto whole=update.optimizer.snapshot();update.optimizer.restore(checkpoint);update.run(4);auto resumed=update.optimizer.snapshot();for(size_t i=0;i<whole.size();++i)check(torch::equal(whole[i],resumed[i]),"resident resume differs");
    // Artificially place the real device counter at its boundaries. No long
    // training run is needed to prove continuation and fail-before-mutation.
    auto set_control=[&](UpdateState s){update.optimizer.control.copy_(torch::from_blob(&s,{int64_t(sizeof(s))},torch::kUInt8));};
    auto control=update.optimizer.state();control.step=1000013;control.segment_start=control.step;set_control(control);update.run(2);check(update.optimizer.state().step==1000015,"optimizer retained obsolete million-update ceiling");
    control=update.optimizer.state();control.step=UINT32_MAX;control.segment_start=control.step;set_control(control);auto before_overflow=update.optimizer.snapshot();update.run(2);auto after_overflow=update.optimizer.snapshot();check(update.optimizer.state().step==UINT32_MAX&&update.optimizer.state().failure==3,"device counter wrapped or failed late");
    for(size_t i=0;i+1<before_overflow.size();++i)check(torch::equal(before_overflow[i],after_overflow[i]),"overflow mutated parameters or Adam moments");
    update.optimizer.restore(initial);{torch::NoGradGuard guard;update.optimizer.parameters[0].fill_(NAN);}auto invalid=update.optimizer.snapshot();update.run(2);check(update.optimizer.state().failure&&update.optimizer.state().step==0,"nonfinite fused update advanced optimizer");
    auto frozen_invalid=update.optimizer.snapshot();for(size_t i=0;i+1<invalid.size();++i)check(torch::allclose(invalid[i],frozen_invalid[i],0,0,true),"nonfinite update mutated parameters or moments");update.optimizer.restore(initial);
    if(backend==UpdateBackend::Fused){
        ActionNetwork reference(placement_schema);reference->to(device);ResidentUpdate oracle(reference,data,31);oracle.optimizer.restore(initial);update.optimizer.restore(initial);
        for(unsigned step=0;step<7;++step){oracle.run(1);update.run(1);check(torch::equal(oracle.ids,update.ids),"fused backend changed sampled IDs");
            check(torch::allclose(oracle.prediction,update.prediction,2e-5,2e-6),"fused forward differs from autograd network");
            for(unsigned i=0;i<6;++i){check(torch::allclose(oracle.optimizer.parameters[i].grad(),update.optimizer.parameters[i].grad(),2e-5,2e-6),"fused gradient differs from autograd");check(torch::allclose(oracle.optimizer.parameters[i],update.optimizer.parameters[i],2e-5,2e-6),"fused update differs from reference");
                check(torch::allclose(oracle.optimizer.mean[i],update.optimizer.mean[i],2e-5,2e-7),"fused Adam first moment differs");check(torch::allclose(oracle.optimizer.variance[i],update.optimizer.variance[i],2e-5,2e-8),"fused Adam second moment differs");}
        }
    }
    auto before=data.root.data_ptr();auto b=page(2,2);data.append(b,"b","second");update.run(1);check(data.root.data_ptr()==before&&update.captures==1,"page append recaptured stable graph");check(update.ids.max().item<int>()>=3,"graph did not observe appended states");
    auto c=page(2,4);data.append(c,"a","first");update.run(1);check(update.pool()==3&&update.captures==2,"capacity growth or unlabeled row compaction");
    // Dense independent oracle: same sampler IDs, expanded bits, partial labels
    // and masks. Removing zero-gradient rows must preserve supervised gradients.
    auto ids=update.ids.cpu();auto input=update.input.cpu();auto labels=update.target.cpu();std::vector<const ActionData*> pages{&a,&b,&c};
    for(unsigned row=0;row<31;++row){unsigned id=unsigned(ids[row].item<int>());const ActionData* selected=nullptr;for(auto* p:pages){if(id<p->states()){selected=p;break;}id-=p->states();}check(selected,"sampler ID outside pages");unsigned out=0;
        for(uint32_t r=selected->offsets[id];r<selected->offsets[id+1];++r){if(!(selected->labels[r]&248))continue;check(labels[row][out].item<int>()==selected->labels[r],"partial label discarded");auto expected=torch::from_blob(const_cast<float*>(selected->x.data())+size_t(r)*placement_features,{placement_features},torch::kFloat32);if(compact)check((input[row][out]-expected).abs().max().item<float>()<=1.f/32767,"compact resident feature exceeded fixed-point rounding bound");else check(torch::equal(input[row][out],expected),"resident feature bits differ");++out;}
        for(;out<update.pool();++out)check(labels[row][out].item<int>()==0&&input[row][out].abs().sum().item<float>()==0,"padding acquired supervision");}
    control=update.optimizer.state();control.step=1000039;control.segment_start=control.step;set_control(control);
    auto saved=update.optimizer.snapshot();auto path=fs::temp_directory_path()/("blitz-resident-checkpoint-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CheckpointWriter writer;auto journal=path/"latest.json";writer.submit(path/"snapshot",model,update.optimizer,{{"test",true}},update.input.flatten(0,1),model->forward(update.input).flatten(0,1),journal,{{"active_training",{{"target_step",update.optimizer.state().step+17}}},{"datasets",{"a","b","c"}}});update.run(2);writer.join();
    auto published=read_json(journal);check(published.at("checkpoint")=="snapshot"&&published.at("checkpoint_sha256")==file_sha256(path/"snapshot/checkpoint.pt"),"recovery journal does not identify verified snapshot");check(published.at("active_training").at("target_step")==update.optimizer.state().step+15,"partial training target lost in journal");
    ActionNetwork restored(placement_schema);restored->to(device);load_checkpoint_model(path/"snapshot/checkpoint.pt",restored,device);auto parameters=restored->parameters();for(size_t i=0;i<parameters.size();++i)check(torch::equal(parameters[i],saved[i]),"warmstart policy differs before first teacher");ResidentUpdate continuation(restored,data,31,{},backend);load_state(path/"snapshot/checkpoint.pt",restored,continuation.optimizer,device);auto frozen=continuation.optimizer.snapshot();for(size_t i=0;i<saved.size();++i)check(torch::equal(saved[i],frozen[i]),"checkpoint writer read mutable GPU state");continuation.run(2);auto x=update.optimizer.snapshot(),y=continuation.optimizer.snapshot();for(size_t i=0;i<x.size();++i)check(torch::equal(x[i],y[i]),"fresh resident checkpoint continuation differs");
    writer.submit(path/"invalid",model,update.optimizer,{{"test",true}},update.input.flatten(0,1),torch::full_like(model->forward(update.input).flatten(0,1),1e6),journal,{{"invalid",true}});bool rejected=false;try{writer.join();}catch(const std::runtime_error&){rejected=true;}check(rejected&&read_json(journal)==published,"failed checkpoint replaced valid recovery pointer");
    {
        std::promise<void> gate;auto release=gate.get_future().share();CheckpointWriter staged(path/"scratch",[release]{release.wait();});
        staged.submit(path/"staged",model,update.optimizer,{{"test",true}},update.input.flatten(0,1),model->forward(update.input).flatten(0,1),journal,{{"local",true}});
        auto local=staged.local_checkpoint();bool isolated=fs::exists(local/"checkpoint.pt")&&!fs::exists(path/"staged")&&read_json(journal)==published;
        update.run(1);gate.set_value();staged.join();check(isolated,"local snapshot waited for or published blocked durable copy");check(read_json(journal).at("checkpoint")=="staged","staged snapshot was not published");
        auto durable=read_json(journal);CheckpointWriter failed(path/"failed-scratch",[]{throw std::runtime_error("injected publication failure");});failed.submit(path/"unpublished",model,update.optimizer,{{"test",true}},{},{},journal,{{"bad",true}});check(!failed.local_checkpoint().empty(),"verified local checkpoint unavailable before publication");rejected=false;try{failed.join();}catch(const std::runtime_error&){rejected=true;}check(rejected&&read_json(journal)==durable,"publication failure replaced valid journal");
    }
    {
        ResidentDataset many(device,739,compact);for(unsigned i=0;i<100;++i)many.append(page(1,2),"asset-"+std::to_string(i),"category-"+std::to_string(i%7));ResidentUpdate sample(model,many,4096,{},backend);sample.gather();auto sampled=sample.ids.cpu();std::set<int> seen;for(unsigned i=0;i<4096;++i)seen.insert(sampled[i].item<int>());check(seen.size()==100,"sampler failed to include 100 independent assets");
        auto stable=many.root.data_ptr();many.clear();many.append(page(3,2),"fresh","fresh-category");sample.run(1);check(many.root.data_ptr()==stable&&sample.ids.max().item<int>()<3,"bounded window retained stale state pointers");
        ResidentDataset limited(device,31,compact,64);rejected=false;try{limited.append(a,"a","first");}catch(const std::length_error&){rejected=true;}check(rejected&&!limited.states(),"dataset budget failure partially published a page");
    }
    if(compact){
        auto exceptions=page(1,2);exceptions.x[0]=1e20f;exceptions.x[1]=1e-20f;exceptions.x[3]=1.01f;exceptions.x[6]=.123456f;
        auto encoded=compact_actions(exceptions);auto expected=expand_compact(encoded);ResidentDataset decoded(device,3,true);decoded.append(encoded,"exceptions","range");ResidentUpdate read(model,decoded,7,{},backend);read.gather();auto input=read.input.cpu();
        for(size_t i=0;i<placement_features;++i)check(std::abs(input[0][0][int64_t(i)].item<float>()-expected.x[i])<=std::abs(expected.x[i])*1e-7f+1e-7f,"GPU compact decoder differs from CPU including escapes");
    }
    ResidentDataset empty_page(device,41,compact);auto empty=page(2,1);std::fill(empty.labels.begin(),empty.labels.end(),0);std::fill(empty.targets.begin(),empty.targets.end(),0);empty_page.append(empty,"empty","none");ResidentUpdate empty_update(restored,empty_page,7,{},backend);empty_update.gather();check(empty_update.input.abs().sum().item<float>()==0&&empty_update.target.sum().item<int>()==0,"unlabeled state produced training rows");fs::remove_all(path);
    std::cout<<json({{"backend",backend==UpdateBackend::Fused?"fused":"reference"},{"epilogues",update.fused_epilogues()},{"compact",compact},{"resident_contracts",true},{"stable_root",true},{"append_without_capture",true},{"partial_labels",true},{"checkpoint_resume_exact",true},{"pool",update.pool()},{"data_bytes",data.bytes()}}).dump()<<'\n';
}
inline void resident_contracts(){
    auto time=CycleTime::duration(2000,120,10);if(time.finishing(121999)||!time.finishing(122000)||time.expired(131999)||!time.expired(132000))throw std::runtime_error("duration boundary changed");
    auto resumed=CycleTime::resume(999000,120,10,75000);if(resumed.finishing(1043999)||!resumed.finishing(1044000)||resumed.expired(1053999)||!resumed.expired(1054000)||resumed.completed(1009000,75000,120)!=85000)throw std::runtime_error("downtime counted as learning");
    auto root=fs::temp_directory_path()/("blitz-history-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(root);uint64_t bytes;
    {CycleHistory h(root/"history.jsonl",0);for(unsigned i=0;i<1000;++i)h.append("phases",{{"iteration",i}});bytes=h.sync();h.append("phases",{{"uncommitted",true}});}
    {CycleHistory h(root/"history.jsonl",bytes);h.append("all_datasets","new");json report;h.assemble(report);if(report.at("phases").size()!=1000||report.at("phases").back().at("iteration")!=999||report.at("all_datasets")!=json::array({"new"}))throw std::runtime_error("history recovery boundary lost or duplicated work");}fs::remove_all(root);
    if(update_target(999999,17)!=1000016||update_target(UINT32_MAX-4,4)!=UINT32_MAX)throw std::runtime_error("long-run counter domain");bool rejected=false;try{update_target(UINT32_MAX-4,5);}catch(const std::overflow_error&){rejected=true;}if(!rejected)throw std::runtime_error("counter overflow wrapped");
    resident_contracts(false);resident_contracts(true);resident_contracts(false,UpdateBackend::Fused);resident_contracts(true,UpdateBackend::Fused);
}
}
