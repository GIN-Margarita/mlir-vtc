#ifndef DIALECT_Vtc_VtcDIALECT_H
#define DIALECT_Vtc_VtcDIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include <cstdint>

namespace mlir {
namespace Vtc {

// Constant dimension identifiers
constexpr static int kIDimension = 0;
constexpr static int kJDimension = 1;
constexpr static int kKDimension = 2;

// Index type size
constexpr static int64_t kIndexSize = 3;

// Index type used to store offsets and bounds
typedef SmallVector<int64_t, kIndexSize> Index;

class VtcDialect : public Dialect {
public:
  explicit VtcDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to Vtc operations
  static StringRef getDialectNamespace() { return "Vtc"; }

  static StringRef getVtcProgramAttrName() { return "Vtc.program"; }

  static StringRef getFieldTypeName() { return "field"; }
  static StringRef getTempTypeName() { return "temp"; }
  static StringRef getResultTypeName() { return "result"; }

//   static bool isVtcProgram(FuncOp funcOp) {
//     return !!funcOp->getAttr(getVtcProgramAttrName());
//   }

//   /// Parses a type registered to this dialect
//   Type parseType(DialectAsmParser &parser) const override;

//   /// Print a type registered to this dialect
//   void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace Vtc
} // namespace mlir

#endif // DIALECT_Vtc_VtcDIALECT_H
