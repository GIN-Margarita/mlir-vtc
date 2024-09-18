#include "Dialect/Vtc/Passes.h"
#include "Dialect/Vtc/VtcDialect.h"
#include "Dialect/Vtc/VtcOps.h"
#include "Dialect/Vtc/VtcTypes.h"
#include "Dialect/Vtc/VtcUtils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

using namespace mlir;
using namespace Vtc;

namespace {

// Base class of all storage materialization patterns
template <typename SourceOp>
struct StorageMaterializationPattern : public OpRewritePattern<SourceOp> {
  StorageMaterializationPattern(MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit) {}

  // Buffer all outputs connected to an apply but not to a store op
  bool doesEdgeRequireBuffering(Value value) const {
    if (llvm::any_of(value.getUsers(),
                     [](Operation *op) { return isa<Vtc::ApplyOp>(op); }) &&
        llvm::none_of(value.getUsers(),
                      [](Operation *op) { return isa<Vtc::StoreOp>(op); }))
      return true;
    return false;
  }

  // Buffer the results of the cloned operation and replace the matched op
  LogicalResult introduceResultBuffers(Operation *matchedOp,
                                       Operation *clonedOp,
                                       PatternRewriter &rewriter) const {
    SmallVector<Value, 10> repResults = clonedOp->getResults();
    for (auto result : matchedOp->getResults()) {
      if (doesEdgeRequireBuffering(result)) {
        auto bufferOp = rewriter.create<Vtc::BufferOp>(
            matchedOp->getLoc(), result.getType(),
            clonedOp->getResult(result.getResultNumber()), nullptr, nullptr);
        repResults[result.getResultNumber()] = bufferOp;
      }
    }
    rewriter.replaceOp(matchedOp, repResults);
    return success();
  }
};

// Pattern introducing buffers between consecutive apply ops
struct ApplyOpRewrite : public StorageMaterializationPattern<Vtc::ApplyOp> {
  using StorageMaterializationPattern<
      Vtc::ApplyOp>::StorageMaterializationPattern;

  LogicalResult matchAndRewrite(Vtc::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(applyOp.getResults(), [&](Value value) {
          return doesEdgeRequireBuffering(value);
        })) {
      // Clone the apply op and move the body
      auto clonedOp = rewriter.cloneWithoutRegions(applyOp);
      rewriter.inlineRegionBefore(applyOp.region(), clonedOp.region(),
                                  clonedOp.region().begin());

      // Introduce a buffer on every result connected to another apply
      introduceResultBuffers(applyOp, clonedOp, rewriter);
      return success();
    }
    return failure();
  }
};

// Pattern introducing buffers between consecutive apply ops
struct CombineOpRewrite
    : public StorageMaterializationPattern<Vtc::CombineOp> {
  using StorageMaterializationPattern<
      Vtc::CombineOp>::StorageMaterializationPattern;

  LogicalResult matchAndRewrite(Vtc::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(combineOp.getResults(), [&](Value value) {
          return doesEdgeRequireBuffering(value);
        })) {
      // Clone the combine op
      auto clonedOp = rewriter.clone(*combineOp.getOperation());

      // Introduce a buffer on every result connected to another apply
      introduceResultBuffers(combineOp, clonedOp, rewriter);
      return success();
    }
    return failure();
  }
};

struct StorageMaterializationPass
    : public StorageMaterializationPassBase<StorageMaterializationPass> {

  void runOnFunction() override;
};

void StorageMaterializationPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as Vtc programs
  if (!VtcDialect::isVtcProgram(funcOp))
    return;

  // Poppulate the pattern list depending on the config
  OwningRewritePatternList patterns;
  patterns.insert<ApplyOpRewrite, CombineOpRewrite>(&getContext());
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createStorageMaterializationPass() {
  return std::make_unique<StorageMaterializationPass>();
}
