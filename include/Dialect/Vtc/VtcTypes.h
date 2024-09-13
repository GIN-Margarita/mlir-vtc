#pragma once 

#include "Dialect/Vtc/VtcDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir
{
namespace Vtc
{
    namespace detail
    {
        struct GridTypeStorage;
        struct FieldTypeStorage;
        struct TempTypeStorage; 
        struct ResultTypeStorage;
    }

class GridType : public Type {
public:
    using ImplType = detail::GridTypeStorage;
    using Type::Type;

    static bool classof(Type type);

    static constexpr int64_t kDynamicDimension = -1;
    static constexpr int64_t kScalarDimension = 0;

    Type getElementType() const;

    ArrayRef<int64_t> getShape() const;

    unsigned getRank() const;

    int64_t hasDynamicShape() const;

    int64_t hasStaticShape() const;

    bool hasEqualShape(ArrayRef<int64_t> lb,ArrayRef<int64_t> ub) const;

    bool hasLargerOrEqualShape(ArrayRef<int64_t> lb,ArrayRef<int64_t> ub) const;    

    SmallVector<bool,3> getAllocation() const;
    
    SmallVector<int64_t,3> getMemRefShape() const;

    static constexpr bool isDynamic(int64_t dimSize)
    {
        return kDynamicDimension == dimSize;
    }

    static constexpr bool isScalar(int64_t dimSize)
    {
        return kScalarDimension == dimSize; 
    }
};

class FieldType : public Type::TypeBase<FieldType, GridType, detail::FieldTypeStorage>  {
public:
    using Base::Base;

    static FieldType get(Type elementType, ArrayRef<int64_t> shape);

};

class TempType : public Type::TypeBase<TempType, GridType, detail::TempTypeStorage>  {

public:
    using Base::Base;

    static TempType get(Type elementType, ArrayRef<int64_t> shape);

    static TempType get(Type elementType, ArrayRef<bool> allocation,ArrayRef<int64_t> lb, ArrayRef<int64_t> ub);

    static TempType get(Type elementType, ArrayRef<bool> allocation);
};
/// Temporaries keep multi-dimensional intermediate results
class ResultType
    : public Type::TypeBase<ResultType, Type, detail::ResultTypeStorage> {
public:
  using Base::Base;

  static ResultType get(Type resultType);

  /// Return the result type
  Type getResultType() const;
};

}
}        // namespace mlir

