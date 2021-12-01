// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>
#include <popart/shapeinference.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <poprand/RandomGen.hpp>
#include <random>

using namespace popart;
using namespace popart::popx;

namespace CustomOperators {
  const popart::OperatorIdentifier FusedReplicatedAllReduce = {"ai.graphcore", "FusedReplicatedAllReduce", 1};
} // namespace CustomOperators

class FusedReplicatedAllReduceOp : public Op {
public:
    CollectiveOperator collective_operator;
    CommGroup group;
    FusedReplicatedAllReduceOp(const OperatorIdentifier &_opid,
	                                             CollectiveOperator op_,
	                                             CommGroup group_,
	                                             const Op::Settings &settings_)
	      : Op(_opid, settings_), collective_operator(op_), group(group_){}

    std::unique_ptr<Op> clone() const override {
     return std::make_unique<FusedReplicatedAllReduceOp>(*this);
    }
    void setup() {}
    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    
    view::Regions modifies(InIndex index) const {
	      return {view::Region::getFull(inShape(index))};
	}
  
	view::Regions aliases(InIndex in, OutIndex out) const {
	  if (in != out) {
      return {view::Region::getEmpty(inRank(in))};
	  } else {
	    return {view::Region::getFull(inShape(in))};
    }
	}
  
  void growAliasModel(AliasModel &m) const override { growAliasModelMulti(m); }	  	
};


class FusedReplicatedAllReduceOpX : public PopOpx
{
public:
    FusedReplicatedAllReduceOpX(popart::Op *op, Devicex *devicex) : PopOpx(op, devicex) {
    verifyOp<FusedReplicatedAllReduceOp>(op, CustomOperators::FusedReplicatedAllReduce);
    }

    InputCreatorType getInputCreatorType(InIndex) const {
	    return InputCreatorType::CanUnwind;
	  }

    snap::Tensor unwindTensorLayout(snap::Tensor tensor, InIndex in, OutIndex out) const {
      if (in == out) {
	      return tensor;
      } else {
        throw error("Unexpected input output pair in FusedReplicatedAllReduce.");
      }  
	  }
	
    view::RegMap unwindRegion(InIndex, OutIndex) const {
      return [](const view::Region &r) { return view::Regions(1, r); };
    }


    void grow(snap::program::Sequence &snap_prog) const final {
      FusedReplicatedAllReduceOp &myOp = getOp<FusedReplicatedAllReduceOp>();

    poplar::program::Sequence &prog = snap_prog.getPoplarSequence();


    // Fill vector of inputs.
    std::vector<snap::Tensor> snap_inputs;
    std::vector<poplar::Tensor> poplar_inputs;
    for (size_t i = 0; i < myOp.input->n(); ++i) {
	    snap_inputs.emplace_back(getInTensor(i));
      poplar_inputs.emplace_back(getInTensor(i).getPoplarTensor());
	  }

    // Call gcl
    poplar::DebugContext debugContext = {};
    const poplar::OptionFlags &allReduceOptions = dv_p->lowering().gclOptions;
    const std::vector<poplar::Tensor> datas{poplar_inputs};
    std::vector<poplar::Tensor> poplar_outputs = gcl::allReduceCrossReplica(graph().getPoplarGraph(), datas,
                                                                getPoplarCollectiveOperator(myOp.collective_operator),
                                                                prog,
                                                                toGCLCommGroup(myOp.group),
                                                                "FusedReplicatedAllReduce",
                                                                allReduceOptions);
    
    // Set outputs
    for (size_t i = 0; i < myOp.input->n(); ++i) {
      if (hasInViewChangers(i)) {
	      setOutViewChangers(i, getInViewChangers(i));
	    }  
	    setOutTensor(i, snap::Tensor{poplar_outputs[i], graph()});
	  }

    }
};

static OpxCreator<FusedReplicatedAllReduceOpX>
    fusedReplicatedAllReduceOpxCreator(CustomOperators::FusedReplicatedAllReduce);

static popart::RegisterShapeInferenceFunction FusedReplicatedAllReduceShapeInfer(
    CustomOperators::FusedReplicatedAllReduce,
            [](ShapeInferenceContext &ctx) {
            for (size_t input_idx=0; input_idx < ctx.getNumOutputs(); input_idx++){
              propagateElemTypeFromInputToOutput(ctx, input_idx, input_idx);
              propagateShapeFromInputToOutput(ctx, input_idx, input_idx);
            }
    });