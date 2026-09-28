#pragma once
#include "blitz/remesher.hpp"
namespace blitz::detail {
struct GenerationHooks {
    std::function<Lod(MeshView,const ReduceSettings&,const EvalSettings&,double)> propose;
    std::function<Measurement(MeshView,MeshView,const Bounds&,const EvalSettings&)> evaluate;
    std::function<bool(Result&)> confirm;
    std::function<double(const Result&)> tiebreak;
};
Result generate_with_hooks(MeshView,const Settings&,const Proposer&,const GenerationHooks*);
}
