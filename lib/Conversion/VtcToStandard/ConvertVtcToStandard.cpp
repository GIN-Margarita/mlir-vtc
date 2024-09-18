#include "Dialect/Vtc/VtcDialect.h"
#include "Conversion/VtcToStandard/ConvertVtcToStandard.h"
#include "Conversion/VtcToStandard/Passes.h"
#include "Dialect/Vtc/VtcOps.h"
#include "Dialect/Vtc/VtcTypes.h"
#include "Dialect/Vtc/VtcUtils.h"
// #include "Dialect/Vtc/ShapeInterface.h"
// #include "Dialect/Vtc/OffsetInterface.h"
// #include "Dialect/Vtc/ShiftInterface.h"
// #include "Dialect/Vtc/ExtentInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>

using namespace mlir;
using namespace Vtc;
using namespace scf;

namespace
{

    //===----------------------------------------------------------------------===//
    // Rewriting Pattern
    //===----------------------------------------------------------------------===//

    class FuncOpLowering : public VtcOpToStdPattern<FuncOp>
    {
    public:
        using VtcOpToStdPattern<FuncOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto funcOp = cast<FuncOp>(operation);

            // Convert the original function arguments
            TypeConverter::SignatureConversion result(funcOp.getNumArguments());
            for (auto &en : llvm::enumerate(funcOp.getType().getInputs()))
                result.addInputs(en.index(), typeConverter.convertType(en.value()));
            auto funcType =
                FunctionType::get(funcOp.getContext(), result.getConvertedTypes(),
                                  funcOp.getType().getResults());

            // Replace the function by a function with an updated signature
            auto newFuncOp =
                rewriter.create<FuncOp>(loc, funcOp.getName(), funcType, llvm::None);
            rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                        newFuncOp.end());

            // Convert the signature and delete the original operation
            rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
            rewriter.eraseOp(funcOp);
            return success();
        }
    };

    class YieldOpLowering : public VtcOpToStdPattern<scf::YieldOp>
    {
    public:
        using VtcOpToStdPattern<scf::YieldOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto yieldOp = cast<scf::YieldOp>(operation);

            // Remove all result types from the operand list
            SmallVector<Value, 4> newOperands;
            llvm::copy_if(
                yieldOp.getOperands(), std::back_inserter(newOperands),
                [](Value value)
                { return !value.getType().isa<ResultType>(); });
            assert(newOperands.size() < yieldOp.getNumOperands() &&
                   "expected if op to return results");

            rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newOperands);
            return success();
        }
    };

    class IfOpLowering : public VtcOpToStdPattern<scf::IfOp>
    {
    public:
        using VtcOpToStdPattern<scf::IfOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto ifOp = cast<scf::IfOp>(operation);

            // Remove all result types from the result list
            SmallVector<Type, 4> newTypes;
            llvm::copy_if(ifOp.getResultTypes(), std::back_inserter(newTypes),
                          [](Type type)
                          { return !type.isa<ResultType>(); });
            assert(newTypes.size() < ifOp.getNumResults() &&
                   "expected if op to return results");

            // Create a new if op and move the bodies
            auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newTypes,
                                                    ifOp.condition(), true);
            newOp.walk([&](scf::YieldOp yieldOp)
                       { rewriter.eraseOp(yieldOp); });
            rewriter.mergeBlocks(ifOp.getBody(0), newOp.getBody(0), llvm::None);
            rewriter.mergeBlocks(ifOp.getBody(1), newOp.getBody(1), llvm::None);

            // Erase the if op if there are no results to replace
            if (newOp.getNumResults() == 0)
            {
                rewriter.eraseOp(ifOp);
                return success();
            }

            // Replace the if op by the results of the new op
            SmallVector<Value, 4> newResults(ifOp.getNumResults(),
                                             newOp.getResults().front());
            auto it = newOp.getResults().begin();
            for (auto en : llvm::enumerate(ifOp.getResultTypes()))
            {
                if (!en.value().isa<ResultType>())
                    newResults[en.index()] = *it++;
            }
            rewriter.replaceOp(ifOp, newResults);
            return success();
        }
    };

    class CastOpLowering : public VtcOpToStdPattern<Vtc::CastOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::CastOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto castOp = cast<Vtc::CastOp>(operation);

            // Compute the static shape of the field and cast the input memref
            auto resType = castOp.res().getType().cast<FieldType>();
            rewriter.replaceOpWithNewOp<MemRefCastOp>(
                operation, operands[0],
                typeConverter.convertType(resType).cast<MemRefType>());
            return success();
        }
    };

    class LoadOpLowering : public VtcOpToStdPattern<Vtc::LoadOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::LoadOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto loadOp = cast<Vtc::LoadOp>(operation);

            // Get the temp and field types
            auto fieldType = loadOp.field().getType().cast<FieldType>();

            // Compute the shape of the subview
            auto subViewShape =
                computeSubViewShape(fieldType, operation, valueToLB[loadOp.field()]);

            // Replace the load op by a subview op
            auto subViewOp = rewriter.create<SubViewOp>(
                loc, operands[0], std::get<0>(subViewShape), std::get<1>(subViewShape),
                std::get<2>(subViewShape));
            rewriter.replaceOp(operation, subViewOp.getResult());
            return success();
        }
    };

    class BufferOpLowering : public VtcOpToStdPattern<Vtc::BufferOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::BufferOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto bufferOp = cast<Vtc::BufferOp>(operation);

            // Free the buffer memory after the last use
            assert(isa<gpu::AllocOp>(operands[0].getDefiningOp()) &&
                   "expected the temporary points to an allocation");
            Operation *lastUser = bufferOp.getOperation();
            for (auto user : bufferOp.getResult().getUsers())
            {
                if (lastUser->isBeforeInBlock(user))
                    lastUser = user;
            }
            rewriter.setInsertionPointAfter(lastUser);
            rewriter.create<gpu::DeallocOp>(loc, TypeRange(),
                                            ValueRange(bufferOp.temp()));

            rewriter.replaceOp(operation, bufferOp.temp());
            return success();
        }
    };

    class ApplyOpLowering : public VtcOpToStdPattern<Vtc::ApplyOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::ApplyOp>::VtcOpToStdPattern;

        // Get the temporary and the shape of the buffer
        std::tuple<Value, mlir::Vtc::ShapeInterface> getShapeAndTemporary(Value value) const
        {
            if (auto storeOp = getUserOp<Vtc::StoreOp>(value))
            {
                return std::make_tuple(storeOp.temp(),
                                       cast<mlir::Vtc::ShapeInterface>(storeOp.getOperation()));
            }
            if (auto bufferOp = getUserOp<Vtc::BufferOp>(value))
            {
                return std::make_tuple(bufferOp.temp(),
                                       cast<mlir::Vtc::ShapeInterface>(bufferOp.getOperation()));
            }
            llvm_unreachable("expected a valid storage operation");
            return std::make_tuple(nullptr, nullptr);
        }

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto applyOp = cast<Vtc::ApplyOp>(operation);
            auto shapeOp = cast<mlir::Vtc::ShapeInterface>(operation);

            // Allocate storage for buffers or introduce get a view of the output field
            SmallVector<Value, 10> newResults;
            for (auto result : applyOp.getResults())
            {
                Value temp;
                mlir::Vtc::ShapeInterface shapeOp;
                std::tie(temp, shapeOp) = getShapeAndTemporary(result);
                auto oldType = temp.getType().cast<TempType>();
                auto tempType =
                    TempType::get(oldType.getElementType(), oldType.getAllocation(),
                                  shapeOp.getLB(), shapeOp.getUB());
                auto allocType = typeConverter.convertType(tempType).cast<MemRefType>();
                assert(allocType.hasStaticShape() &&
                       "expected buffer to have a static shape");
                auto segAttr = rewriter.getNamedAttr(
                    "operand_segment_sizes", rewriter.getI32VectorAttr({0, 0, 0}));
                auto allocOp = rewriter.create<gpu::AllocOp>(loc, TypeRange(allocType),
                                                             ValueRange(), segAttr);
                newResults.push_back(allocOp.getResult(0));
            }

            // Compute the loop bounds starting from zero
            // (in case of loop unrolling adjust the step of the loop)
            SmallVector<Value, 3> lbs, ubs, steps;
            auto returnOp = cast<Vtc::ReturnOp>(applyOp.getBody()->getTerminator());
            for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i)
            {
                int64_t lb = shapeOp.getLB()[i];
                int64_t ub = shapeOp.getUB()[i];
                int64_t step = returnOp.unroll().hasValue() ? returnOp.getUnroll()[i] : 1;
                lbs.push_back(rewriter.create<ConstantIndexOp>(loc, lb));
                ubs.push_back(rewriter.create<ConstantIndexOp>(loc, ub));
                steps.push_back(rewriter.create<ConstantIndexOp>(loc, step));
            }

            // Convert the signature of the apply op body
            // (access the apply op operands and introduce the loop indicies)
            TypeConverter::SignatureConversion result(applyOp.getNumOperands());
            for (auto &en : llvm::enumerate(applyOp.getOperands()))
            {
                result.remapInput(en.index(), operands[en.index()]);
            }
            rewriter.applySignatureConversion(&applyOp.region(), result);

            // Affine map used for induction variable computation
            // TODO this is only useful for sequential loops
            auto fwdExpr = rewriter.getAffineDimExpr(0);
            auto fwdMap = AffineMap::get(1, 0, fwdExpr);

            // Replace the Vtc apply operation by a loop nest
            auto parallelOp = rewriter.create<ParallelOp>(loc, lbs, ubs, steps);
            rewriter.mergeBlockBefore(
                applyOp.getBody(),
                parallelOp.getLoopBody().getBlocks().back().getTerminator());

            // Insert index variables at the beginning of the loop body
            rewriter.setInsertionPointToStart(parallelOp.getBody());
            for (int64_t i = 0, e = shapeOp.getRank(); i != e; ++i)
            {
                rewriter.create<AffineApplyOp>(
                    loc, fwdMap, ValueRange(parallelOp.getInductionVars()[i]));
            }

            // Replace the applyOp
            rewriter.replaceOp(applyOp, newResults);
            return success();
        }
    }; // namespace

    class StoreResultOpLowering
        : public VtcOpToStdPattern<Vtc::StoreResultOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::StoreResultOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto resultOp = cast<Vtc::StoreResultOp>(operation);

            // Iterate over all return op operands of the result
            for (auto opOperand : valueToReturnOpOperands[resultOp.res()])
            {
                // Get the return op and the parallel op
                auto returnOp = cast<Vtc::ReturnOp>(opOperand->getOwner());
                auto parallelOp = returnOp->getParentOfType<ParallelOp>();

                // Check the parent has been lowered
                if (!isa<ParallelOp>(returnOp->getParentOp()))
                    return failure();

                // Store the result in case there is something to store
                if (resultOp.operands().size() == 1)
                {
                    // Compute unroll factor and dimension
                    auto unrollFac = returnOp.getUnrollFac();
                    size_t unrollDim = returnOp.getUnrollDim();

                    // Get the output buffer
                    gpu::AllocOp allocOp;
                    unsigned bufferCount = (returnOp.getNumOperands() / unrollFac) -
                                           (opOperand->getOperandNumber() / unrollFac);
                    auto *node = parallelOp.getOperation();
                    while (bufferCount != 0 && (node = node->getPrevNode()))
                    {
                        if ((allocOp = dyn_cast<gpu::AllocOp>(node)))
                            bufferCount--;
                    }
                    assert(bufferCount == 0 && "expected valid buffer allocation");

                    // Compute the static store offset
                    auto lb = valueToLB[opOperand->get()];
                    llvm::transform(lb, lb.begin(), std::negate<int64_t>());
                    lb[unrollDim] += opOperand->getOperandNumber() % unrollFac;

                    // Set the insertion point to the defining op if possible
                    auto result = resultOp.operands().front();
                    if (result.getDefiningOp() &&
                        result.getDefiningOp()->getParentOp() == resultOp->getParentOp())
                        rewriter.setInsertionPointAfter(result.getDefiningOp());

                    // Compute the index values and introduce the store operation
                    auto inductionVars = getInductionVars(operation);
                    SmallVector<bool, 3> allocation(lb.size(), true);
                    auto storeOffset =
                        computeIndexValues(inductionVars, lb, allocation, rewriter);
                    rewriter.create<mlir::StoreOp>(loc, result, allocOp.getResult(0),
                                                   storeOffset);
                }
            }

            rewriter.eraseOp(operation);
            return success();
        }
    };

    class ReturnOpLowering : public VtcOpToStdPattern<Vtc::ReturnOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::ReturnOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            rewriter.eraseOp(operation);
            return success();
        }
    };

    class AccessOpLowering : public VtcOpToStdPattern<Vtc::AccessOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::AccessOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto accessOp = cast<Vtc::AccessOp>(operation);
            auto offsetOp = cast<mlir::Vtc::OffsetInterface>(accessOp.getOperation());

            // Get the induction variables
            auto inductionVars = getInductionVars(operation);
            if (inductionVars.size() == 0)
                return failure();
            assert(inductionVars.size() == offsetOp.getOffset().size() &&
                   "expected loop nest and access offset to have the same size");

            // Add the lower bound of the temporary to the access offset
            auto totalOffset =
                applyFunElementWise(offsetOp.getOffset(), valueToLB[accessOp.temp()],
                                    std::minus<int64_t>());
            auto tempType = accessOp.temp().getType().cast<TempType>();
            auto loadOffset = computeIndexValues(inductionVars, totalOffset,
                                                 tempType.getAllocation(), rewriter);

            // Replace the access op by a load op
            rewriter.replaceOpWithNewOp<mlir::LoadOp>(operation, operands[0],
                                                      loadOffset);
            return success();
        }
    };

    class DynAccessOpLowering : public VtcOpToStdPattern<Vtc::DynAccessOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::DynAccessOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto dynAccessOp = cast<Vtc::DynAccessOp>(operation);

            // Get the induction variables
            auto inductionVars = getInductionVars(operation);
            if (inductionVars.size() == 0)
                return failure();
            assert(inductionVars.size() == dynAccessOp.offset().size() &&
                   "expected loop nest and access offset to have the same size");

            // Add the negative lower bound to the offset
            auto tempType = dynAccessOp.temp().getType().cast<TempType>();
            auto tempLB = valueToLB[dynAccessOp.temp()];
            llvm::transform(tempLB, tempLB.begin(), std::negate<int64_t>());
            auto loadOffset = computeIndexValues(dynAccessOp.offset(), tempLB,
                                                 tempType.getAllocation(), rewriter);

            // Replace the access op by a load op
            rewriter.replaceOpWithNewOp<mlir::LoadOp>(operation, operands[0],
                                                      loadOffset);
            return success();
        }
    };

    class IndexOpLowering : public VtcOpToStdPattern<Vtc::IndexOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::IndexOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto indexOp = cast<Vtc::IndexOp>(operation);
            auto offsetOp = cast<mlir::Vtc::OffsetInterface>(indexOp.getOperation());

            // Get the induction variables
            auto inductionVars = getInductionVars(operation);
            if (inductionVars.size() == 0)
                return failure();
            assert(inductionVars.size() == offsetOp.getOffset().size() &&
                   "expected loop nest and access offset to have the same size");

            // Shift the induction variable by the offset
            auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
            auto map = AffineMap::get(2, 0, expr);
            SmallVector<Value, 2> params = {
                inductionVars[indexOp.dim()],
                rewriter
                    .create<ConstantIndexOp>(loc, offsetOp.getOffset()[indexOp.dim()])
                    .getResult()};

            // replace the index ob by an affine apply op
            rewriter.replaceOpWithNewOp<mlir::AffineApplyOp>(operation, map, params);

            return success();
        }
    };

    class StoreOpLowering : public VtcOpToStdPattern<Vtc::StoreOp>
    {
    public:
        using VtcOpToStdPattern<Vtc::StoreOp>::VtcOpToStdPattern;

        LogicalResult
        matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            auto loc = operation->getLoc();
            auto storeOp = cast<Vtc::StoreOp>(operation);

            // Get the temp and field types
            auto fieldType = storeOp.field().getType().cast<FieldType>();

            // Compute the shape of the subview
            auto subViewShape =
                computeSubViewShape(fieldType, operation, valueToLB[storeOp.field()]);

            // Replace the allocation by a subview
            auto allocOp = operands[0].getDefiningOp();
            rewriter.setInsertionPoint(allocOp);
            auto subViewOp = rewriter.create<SubViewOp>(
                loc, operands[1], std::get<0>(subViewShape), std::get<1>(subViewShape),
                std::get<2>(subViewShape));
            rewriter.replaceOp(allocOp, subViewOp.getResult());
            rewriter.eraseOp(operation);
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // Conversion Target
    //===----------------------------------------------------------------------===//

    class VtcToStdTarget : public ConversionTarget
    {
    public:
        explicit VtcToStdTarget(MLIRContext &context)
            : ConversionTarget(context) {}

        bool isDynamicallyLegal(Operation *op) const override
        {
            if (auto funcOp = dyn_cast<FuncOp>(op))
            {
                return !VtcDialect::isVtcProgram(funcOp);
            }
            if (auto ifOp = dyn_cast<scf::IfOp>(op))
            {
                return llvm::none_of(ifOp.getResultTypes(),
                                     [](Type type)
                                     { return type.isa<ResultType>(); });
            }
            if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
            {
                return llvm::none_of(yieldOp.getOperandTypes(),
                                     [](Type type)
                                     { return type.isa<ResultType>(); });
            }
            return true;
        }
    };

    //===----------------------------------------------------------------------===//
    // Rewriting Pass
    //===----------------------------------------------------------------------===//

    struct VtcToStandardPass
        : public VtcToStandardPassBase<VtcToStandardPass>
    {
        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<AffineDialect>();
        }
        void runOnOperation() override;

    private:
        Index findLB(Value value);
    };

    Index VtcToStandardPass::findLB(Value value)
    {
        SmallVector<Operation *> operations(value.getUsers().begin(),
                                            value.getUsers().end());
        if (auto definingOp = value.getDefiningOp())
            operations.push_back(definingOp);
        // Search the lower bound of the value
        for (auto op : operations)
        {
            if (auto loadOp = dyn_cast<Vtc::LoadOp>(op))
                return cast<mlir::Vtc::ShapeInterface>(loadOp.getOperation()).getLB();
            if (auto storeOp = dyn_cast<Vtc::StoreOp>(op))
                return cast<mlir::Vtc::ShapeInterface>(storeOp.getOperation()).getLB();
            if (auto bufferOp = dyn_cast<Vtc::BufferOp>(op))
                return cast<mlir::Vtc::ShapeInterface>(bufferOp.getOperation()).getLB();
        }
        return {};
    }

    void VtcToStandardPass::runOnOperation()
    {
        OwningRewritePatternList patterns;
        auto module = getOperation();

        // Check all shapes are set
        auto shapeResult = module.walk([&](mlir::Vtc::ShapeInterface shapeOp)
                                       {
    if (!shapeOp.hasShape()) {
      shapeOp.emitOpError("expected to have a valid shape");
      return WalkResult::interrupt();
    }
    return WalkResult::advance(); });
        if (shapeResult.wasInterrupted())
        {
            return signalPassFailure();
        }

        // Store the input bounds of the Vtc program
        DenseMap<Value, Index> valueToLB;
        module.walk([&](Vtc::CastOp castOp)
                    { valueToLB[castOp.res()] = cast<mlir::Vtc::ShapeInterface>(castOp.getOperation()).getLB(); });
        module.walk([&](Vtc::ApplyOp applyOp)
                    {
    auto shapeOp = cast<mlir::Vtc::ShapeInterface>(applyOp.getOperation());
    // Store the lower bounds for all arguments
    for (auto en : llvm::enumerate(applyOp.getOperands())) {
      valueToLB[applyOp.getBody()->getArgument(en.index())] =
          findLB(en.value());
    }
    // Store the lower bounds for all results
    auto returnOp = cast<Vtc::ReturnOp>(applyOp.getBody()->getTerminator());
    for (auto en : llvm::enumerate(applyOp.getResults())) {
      Index lb = findLB(en.value());
      assert(lb.size() == shapeOp.getRank() &&
             "expected to find valid storage shape");
      // Store the bound for all return op operands writting to the result
      unsigned unrollFac = returnOp.getUnrollFac();
      for (unsigned i = 0, e = unrollFac; i != e; ++i) {
        valueToLB[returnOp.getOperand(en.index() * unrollFac + i)] = lb;
      }
    } });

        // Check there is exactly one storage operation per apply op result
        auto uniqueStorageResult = module.walk([&](Vtc::ApplyOp applyOp)
                                               {
    for (auto result : applyOp.getResults()) {
      unsigned storageOps = 0;
      for (auto user : result.getUsers()) {
        if (isa<Vtc::BufferOp>(user) || isa<Vtc::StoreOp>(user)) {
          storageOps++;
        }
      }
      if (storageOps != 1) {
        applyOp.emitOpError("expected apply op results to have storage");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance(); });
        if (uniqueStorageResult.wasInterrupted())
            return signalPassFailure();

        // Store the return op operands for the result values
        DenseMap<Value, SmallVector<OpOperand *, 10>> valueToReturnOpOperands;
        auto storeMappingResult = module.walk([&](Vtc::StoreResultOp resultOp)
                                              {
    if (!resultOp.getReturnOpOperands()) {
      resultOp.emitOpError("expected valid return op operands");
      return WalkResult::interrupt();
    }
    valueToReturnOpOperands[resultOp.res()] =
        resultOp.getReturnOpOperands().getValue();
    return WalkResult::advance(); });
        if (storeMappingResult.wasInterrupted())
            return signalPassFailure();

        VtcTypeConverter typeConverter(module.getContext());
        populateVtcToStdConversionPatterns(typeConverter, valueToLB,
                                           valueToReturnOpOperands, patterns);

        VtcToStdTarget target(*(module.getContext()));
        target.addLegalDialect<AffineDialect>();
        target.addLegalDialect<StandardOpsDialect>();
        target.addLegalDialect<SCFDialect>();
        target.addDynamicallyLegalOp<FuncOp>();
        target.addDynamicallyLegalOp<scf::IfOp>();
        target.addDynamicallyLegalOp<scf::YieldOp>();
        target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
        target.addLegalOp<gpu::AllocOp>();
        target.addLegalOp<gpu::DeallocOp>();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
        {
            signalPassFailure();
        }
    }

} // namespace

// namespace mlir
// {
//     namespace Vtc
//     {

// Populate the conversion pattern list
void populateVtcToStdConversionPatterns(
    VtcTypeConverter &typeConveter, DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
    mlir::OwningRewritePatternList &patterns)
{
    patterns.insert<FuncOpLowering, IfOpLowering, YieldOpLowering, CastOpLowering,
                    LoadOpLowering, ApplyOpLowering, BufferOpLowering,
                    ReturnOpLowering, StoreResultOpLowering, AccessOpLowering,
                    DynAccessOpLowering, IndexOpLowering, StoreOpLowering>(
        typeConveter, valueToLB, valueToReturnOpOperands);
}

//===----------------------------------------------------------------------===//
// Vtc Type Converter
//===----------------------------------------------------------------------===//

VtcTypeConverter::VtcTypeConverter(MLIRContext *context_)
    : context(context_)
{
    // Add a type conversion for the Vtc field type
    addConversion([&](GridType type)
                  { return MemRefType::get(type.getMemRefShape(), type.getElementType()); });
    addConversion([&](Type type) -> Optional<Type>
                  {
    if (auto gridType = type.dyn_cast<GridType>())
      return llvm::None;
    return type; });
}

//===----------------------------------------------------------------------===//
// Vtc Pattern Base Class
//===----------------------------------------------------------------------===//

VtcToStdPattern::VtcToStdPattern(
    StringRef rootOpName, VtcTypeConverter &typeConverter,
    DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
    PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter.getContext()),
      typeConverter(typeConverter), valueToLB(valueToLB),
      valueToReturnOpOperands(valueToReturnOpOperands) {}

mlir::Vtc::Index VtcToStdPattern::computeShape(mlir::Vtc::ShapeInterface shapeOp) const
{
    return applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(),
                               std::minus<int64_t>());
}

SmallVector<Value, 3>
VtcToStdPattern::getInductionVars(Operation *operation) const
{
    SmallVector<Value, 3> inductionVariables;

    // Get the parallel loop
    auto parallelOp = operation->getParentOfType<ParallelOp>();
    // TODO only useful for sequential applies
    auto forOp = operation->getParentOfType<ForOp>();
    if (!parallelOp)
        return inductionVariables;

    // Collect the induction variables
    parallelOp.walk([&](AffineApplyOp applyOp)
                    {
    for (auto operand : applyOp.getOperands()) {
      // TODO only useful for sequential applies
      if (forOp && forOp.getInductionVar() == operand) {
        inductionVariables.push_back(applyOp.getResult());
        break;
      }
      if (llvm::is_contained(parallelOp.getInductionVars(), operand)) {
        inductionVariables.push_back(applyOp.getResult());
        break;
      }
    } });
    return inductionVariables;
}

std::tuple<mlir::Vtc::Index, mlir::Vtc::Index, mlir::Vtc::Index>
VtcToStdPattern::computeSubViewShape(FieldType fieldType, ShapeInterface shapeOp,
                                     Index castLB) const
{
    auto shape = computeShape(shapeOp);
    Index revShape, revOffset, revStrides;
    for (auto en : llvm::enumerate(fieldType.getAllocation()))
    {
        // Insert values at the front to convert from column- to row-major
        if (en.value())
        {
            revShape.insert(revShape.begin(), shape[en.index()]);
            revStrides.insert(revStrides.begin(), 1);
            revOffset.insert(revOffset.begin(),
                             shapeOp.getLB()[en.index()] - castLB[en.index()]);
        }
    }
    return std::make_tuple(revOffset, revShape, revStrides);
}

SmallVector<Value, 3> VtcToStdPattern::computeIndexValues(
    ValueRange inductionVars, Index offset, ArrayRef<bool> allocation,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = rewriter.getInsertionPoint()->getLoc();
    auto expr = rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
    auto map = AffineMap::get(2, 0, expr);
    SmallVector<Value, 3> resOffset;
    for (auto en : llvm::enumerate(allocation))
    {
        // Insert values at the front to convert from column- to row-major
        if (en.value())
        {
            SmallVector<Value, 2> params = {
                inductionVars[en.index()],
                rewriter.create<ConstantIndexOp>(loc, offset[en.index()])
                    .getResult()};
            auto affineApplyOp = rewriter.create<AffineApplyOp>(loc, map, params);
            resOffset.insert(resOffset.begin(), affineApplyOp.getResult());
        }
    }
    return resOffset;
}

// } // namespace Vtc
// } // namespace mlir

std::unique_ptr<Pass> mlir::createConvertVtcToStandardPass()
{
    return std::make_unique<VtcToStandardPass>();
}
