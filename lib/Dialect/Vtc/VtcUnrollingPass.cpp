#include "Dialect/Vtc/Passes.h"
#include "Dialect/Vtc/VtcDialect.h"
#include "Dialect/Vtc/VtcOps.h"
#include "Dialect/Vtc/VtcTypes.h"
#include "Dialect/Vtc/VtcUtils.h"
// #include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace Vtc;

namespace {

struct VtcUnrollingPass
    : public VtcUnrollingPassBase<VtcUnrollingPass> {

  void runOnFunction() override;

protected:
  void unrollVtcApply(Vtc::ApplyOp applyOp);
  void addPeelIteration(Vtc::ApplyOp applyOp);

  void makePeelIteration(Vtc::ReturnOp returnOp, unsigned tripCount);
  Vtc::ReturnOp cloneBody(Vtc::ApplyOp from, Vtc::ApplyOp to,
                              OpBuilder &builder);
};

Vtc::ReturnOp VtcUnrollingPass::cloneBody(Vtc::ApplyOp from,
                                                  Vtc::ApplyOp to,
                                                  OpBuilder &builder) {
  // Setup the argument mapper
  BlockAndValueMapping mapper;
  for (auto it : llvm::zip(from.getBody()->getArguments(),
                           to.getBody()->getArguments())) {
    mapper.map(std::get<0>(it), std::get<1>(it));
  }
  // Clone the apply op body
  Operation *last = nullptr;
  for (auto &op : from.getBody()->getOperations()) {
    last = builder.clone(op, mapper);
  }
  return cast<Vtc::ReturnOp>(last);
}

void VtcUnrollingPass::unrollVtcApply(Vtc::ApplyOp applyOp) {
  // Setup the builder and
  OpBuilder b(applyOp);

  // Prepare a clone containing a single iteration and an argument mapper
  auto clonedOp = applyOp.clone();

  // Keep a list of the return ops for all unrolled loop iterations
  SmallVector<Vtc::ReturnOp, 4> loopIterations = {
      cast<Vtc::ReturnOp>(applyOp.getBody()->getTerminator())};

  // Keep unrolling until there is one returnOp for every iteration
  b.setInsertionPointToEnd(applyOp.getBody());
  while (loopIterations.size() < unrollFactor) {
    // Update the offsets of the clone
    clonedOp.getBody()->walk([&](Vtc::ShiftInterface shiftOp) {
      Index offset(kIndexSize, 0);
      offset[unrollIndex] = 1;
      shiftOp.shiftByOffset(offset);
    });
    // Clone the body and store the return op
    loopIterations.push_back(cloneBody(clonedOp, applyOp, b));
  }
  clonedOp.erase();

  // Compute the results for the unrolled apply op
  SmallVector<Value, 16> newResults;
  for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
    llvm::transform(
        loopIterations, std::back_inserter(newResults),
        [&](Vtc::ReturnOp returnOp) { return returnOp.getOperand(i); });
  }
  for (auto returnOp : loopIterations) {
    returnOp.erase();
  }

  // Create a new return op returning all results
  SmallVector<int64_t, kIndexSize> unroll(kIndexSize, 1);
  unroll[unrollIndex] = unrollFactor;
  b.create<Vtc::ReturnOp>(loopIterations.front().getLoc(), newResults,
                              b.getI64ArrayAttr(unroll));
}

void VtcUnrollingPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  // Only run on functions marked as Vtc programs
  if (!VtcDialect::isVtcProgram(funcOp))
    return;

  // Check for valid unrolling indexes
  if (unrollIndex == 0) {
    funcOp.emitError("unrolling the innermost loop is not supported");
    signalPassFailure();
    return;
  }

  // Collect the Vtc apply operations
  SmallVector<Vtc::ApplyOp, 16> workList;
  funcOp.walk([&](Vtc::ApplyOp applyOp) { workList.push_back(applyOp); });

  // Unroll the Vtc apply operations
  for (auto applyOp : workList) {
    unrollVtcApply(applyOp);
  }
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createVtcUnrollingPass() {
  return std::make_unique<VtcUnrollingPass>();
}
