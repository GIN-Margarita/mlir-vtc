#ifndef DIALECT_Vtc_VtcUTILS_H
#define DIALECT_Vtc_VtcUTILS_H

#include "Dialect/Vtc/VtcDialect.h"
#include "Dialect/Vtc/VtcTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <functional>

namespace mlir {
namespace Vtc {

/// Helper method that computes the minimum and maximum index
int64_t min(int64_t x, int64_t y);
int64_t max(int64_t x, int64_t y);

/// Helper method to compute element-wise index computations
Index applyFunElementWise(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                          std::function<int64_t(int64_t, int64_t)> fun);

/// Helper to detect an optional array attribute
template <typename T>
bool isOptionalArrayAttr(T x) {
  return std::is_same<T, Optional<ArrayAttr>>::value;
}

} // namespace Vtc
} // namespace mlir

#endif // DIALECT_Vtc_VtcUTILS_H
