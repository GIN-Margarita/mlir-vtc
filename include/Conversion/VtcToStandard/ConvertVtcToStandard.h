#pragma once

#include "Dialect/Vtc/ShapeInterface.h"
#include "Dialect/Vtc/VtcDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
// #include "Dialect/Vtc/OffsetInterface.h"
// #include "Dialect/Vtc/ShiftInterface.h"
// #include "Dialect/Vtc/ExtentInterface.h"
#include <cstdint>
#include <tuple>
using namespace mlir;
using namespace Vtc;
// namespace mlir
// {
//     namespace Vtc
//     {

struct VtcTypeConverter : public TypeConverter
{
    using TypeConverter::TypeConverter;

    VtcTypeConverter(MLIRContext *context);

    MLIRContext *getContext() const { return context; }

private:
    MLIRContext *context;
};

class VtcToStdPattern : public ConversionPattern
{
public:
    VtcToStdPattern(
        StringRef rootOpName, VtcTypeConverter &typeConverter,
        DenseMap<Value, Index> &valueToLB,
        DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
        PatternBenefit benefit = 1);

    // Return the induction variables of the parent loop nest
    SmallVector<Value, 3> getInductionVars(Operation *operation) const;

    /// Compute the shape of the operation
    mlir::Vtc::Index computeShape(ShapeInterface shapeOp) const;

    /// Compute offset, shape, strides of the subview
    std::tuple<Index, Index, Index> computeSubViewShape(FieldType fieldType,
                                                        ShapeInterface accessOp,
                                                        Index assertLB) const;

    /// Compute the index values for a given constant offset
    SmallVector<Value, 3>
    computeIndexValues(ValueRange inductionVars, Index offset,
                       ArrayRef<bool> allocation,
                       ConversionPatternRewriter &rewriter) const;

    /// Return operation of a specific type that uses a given value
    template <typename OpTy>
    OpTy getUserOp(Value value) const
    {
        for (auto user : value.getUsers())
            if (OpTy op = dyn_cast<OpTy>(user))
                return op;
        return nullptr;
    }

protected:
    /// Reference to the type converter
    VtcTypeConverter &typeConverter;

    /// Map storing the lower bounds of the original program
    DenseMap<Value, Index> &valueToLB;

    /// Map the result values to the return op operand
    DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands;
};

template <typename OpTy>
class VtcOpToStdPattern : public VtcToStdPattern
{
public:
    VtcOpToStdPattern(
        VtcTypeConverter &typeConverter, DenseMap<Value, Index> &valueToLB,
        DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
        PatternBenefit benefit = 1)
        : VtcToStdPattern(OpTy::getOperationName(), typeConverter, valueToLB,
                          valueToReturnOpOperands, benefit) {}
};

/// Helper method to populate the conversion pattern list
void populateVtcToStdConversionPatterns(
    VtcTypeConverter &typeConveter, DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, SmallVector<OpOperand *, 10>> &valueToReturnOpOperands,
    OwningRewritePatternList &patterns);
//     } // namespace vtc

// } // namespace mlir
