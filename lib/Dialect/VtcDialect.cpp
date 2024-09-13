#include "Dialect/Vtc/VtcDialect.h"
#include "Dialect/Vtc/VtcOps.h"
// #include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::Vtc;

//===----------------------------------------------------------------------===//
// Vtc Dialect
//===----------------------------------------------------------------------===//

VtcDialect::VtcDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VtcDialect>()) {
//   addTypes<FieldType, TempType, ResultType>();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Vtc/VtcOps.cpp.inc"
      >();
      
  // Allow Vtc operations to exist in their generic form
  allowUnknownOperations();
}
