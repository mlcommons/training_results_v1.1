// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fused_replicated_allreduce.cpp"
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>

class ZFusedReplicatedAllReducePattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        auto &ir = op->getIr();
        // Don't run in inference
        if (!ir.canTrain()) {
            return false;
        }

        // In order for the all-reduces to exist, the optimizer must have been decomposed
        if (!ir.hasDecomposedOptimizers()) {
            return false;
        }
        auto reduce = dynamic_cast<popart::ReplicatedAllReduceOp *>(op);
        if (reduce) {
            bool is_grad_reduce = reduce->inId(reduce->getInIndex()).find(popart::reservedAccumPrefix()) != std::string::npos;
            return is_grad_reduce;
        }

        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *op) const override {
        auto &graph = op->getGraph();

        std::vector<popart::OpId> reduce_ops;
        for (auto &element : graph.getOps()){
            popart::OpId op_id = element.first;
            std::unique_ptr<popart::Op> &candidate = element.second;
            auto reduce = dynamic_cast<popart::ReplicatedAllReduceOp *>(candidate.get());
            if (reduce){
                bool is_grad_reduce = reduce->inId(reduce->getInIndex()).find(popart::reservedAccumPrefix()) != std::string::npos;
                if (is_grad_reduce){
                    reduce_ops.emplace_back(op_id);
                }
            }
        }
        
        // Go through the reduce ops
        std::vector<const popart::Tensor*> touched_tensors;
        for (auto old_reduce_id : reduce_ops){
            popart::Op* old_reduce = graph.getOp(old_reduce_id);
            touched_tensors.emplace_back(old_reduce->outTensor(0));
        }
        return touched_tensors;
    }

    bool apply(popart::Op *op) const override {
        // Get the graph and look through all Ops to identify all that are reduce ops
        auto &graph = op->getGraph();
        // VGraphId vid = op->getVirtualGraphId();
        
        std::map<VGraphId, std::vector<popart::OpId>> reduce_ops;
        for (auto &element : graph.getOps()){
            popart::OpId op_id = element.first;
            std::unique_ptr<popart::Op> &candidate = element.second;
            auto reduce = dynamic_cast<popart::ReplicatedAllReduceOp *>(candidate.get());
            if (reduce){
                bool is_grad_reduce = reduce->inId(reduce->getInIndex()).find(popart::reservedAccumPrefix()) != std::string::npos;
                if (is_grad_reduce){
                    reduce_ops[reduce->getVirtualGraphId()].emplace_back(op_id);
                }
            }
        }

        // Create one allreduce op per virtual graph
        for (auto &element : reduce_ops){
            VGraphId vid = element.first;
            std::vector<popart::OpId> &reduce_ops_in_vgraph = element.second;
            popart::logging::info("Creating FusedReplicatedAllReduce for virtualGraph {}", vid);
            popart::logging::info("VirtualGraph {} contains ReplicatedAllReduce ops: {}", vid, reduce_ops_in_vgraph);

            // Go through all the reduce ops in the virtual graph
            std::vector<popart::Tensor*> inputs;
            std::vector<popart::Tensor*> outputs;
            popart::ReplicatedAllReduceOp* old_reduce;
            for (auto old_reduce_id : reduce_ops_in_vgraph){
                old_reduce = dynamic_cast<popart::ReplicatedAllReduceOp *>(graph.getOp(old_reduce_id));
                
                // Detach inputs and collect them
                inputs.emplace_back(old_reduce->inTensor(old_reduce->getInIndex()));
                old_reduce->disconnectAllInputs();

                // Detach outputs and collect them
                outputs.emplace_back(old_reduce->outTensor(old_reduce->getOutIndex()));
                old_reduce->disconnectAllOutputs();

                // Remove old ops
                popart::logging::info("Erased reduce op {}", old_reduce->str());
                graph.eraseOp(old_reduce_id);
            }

            // Connect the inputs to the fused all reduce
            popart::ReplicatedAllReduceOp *matched_allreduce = dynamic_cast<popart::ReplicatedAllReduceOp *>(old_reduce); 
            CollectiveOperator collective_type = matched_allreduce->getCollectiveOp();
            CommGroup group = matched_allreduce->getGCLCommGroup();
        
            // Fused reduce op that will be used to replace all other reduces
            auto fused_reduce_op = std::make_unique<FusedReplicatedAllReduceOp>(
                                                    CustomOperators::FusedReplicatedAllReduce,
                                                    collective_type, group,
                                                    popart::Op::Settings(graph, "FusedReplicatedAllReduce"));
            fused_reduce_op->setVirtualGraphId(vid);

            
            size_t input_index{0};
            for (auto &input_tensor : inputs){
                fused_reduce_op->connectInTensor(input_index, input_tensor->id);
                input_index++;
            }

            // Connect the outputs
            size_t output_index{0};
            for (auto &output_tensor : outputs){
                fused_reduce_op->connectOutTensor(output_index, output_tensor->id);
                output_index++;
            }

            // Move into graph
            fused_reduce_op->settings.executionContext = ExecutionContext::AccumulateOuterFragment;
            graph.moveIntoGraph(std::move(fused_reduce_op));
        }

        return true;
    }
};

static popart::PatternCreator<ZFusedReplicatedAllReducePattern> ZFusedReplicatedAllReducePatternCreator("ZFusedReplicatedAllReducePattern", false);
