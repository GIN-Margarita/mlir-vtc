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

Type VtcDialect::parseType(DialectAsmParser &parser) const {
    StringRef prefix;

    if(parser.parseKeyword(&prefix))
    {
        parser.emitError(parser.getNameLoc(), "expected type identifier");
        return Type();
    }

    if(prefix == getResultTypeName())
        {
                Type resultType ;
                if(parser.parseLess() || parser.parseType(resultType) || parser.parseGreater())
                {
                    parser.emitError(parser.getNameLoc(), "expected <type>");
                    return Type();
                }
            return ResultType::get(resultType);
        }
     // Parse a field or temp type
  if (prefix == getFieldTypeName() || prefix == getTempTypeName()) {
    // Parse the shape
    SmallVector<int64_t, 3> shape;
    if (parser.parseLess() || parser.parseDimensionList(shape)) {
      parser.emitError(parser.getNameLoc(), "expected valid dimension list");
      return Type();
    }

    // Parse the element type
    Type elementType;
    if (parser.parseType(elementType) || parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "expected valid element type");
      return Type();
    }

    // Return the Stencil type
    if (prefix == getFieldTypeName())
      return FieldType::get(elementType, shape);
    else
      return TempType::get(elementType, shape);
  }

  // Failed to parse a stencil type
  parser.emitError(parser.getNameLoc(), "unknown stencil type ")
      << parser.getFullSymbolSpec();
  return Type();
}


//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

namespace {

void printGridOrTempType(StringRef name, Type type, DialectAsmPrinter &printer) {
  printer << name;
  printer << "<";
  for (auto size : type.cast<GridType>().getShape()) {
    if (size == GridType::kDynamicDimension)
      printer << "?";
    else
      printer << size;
    printer << "x";
  }
  printer << type.cast<GridType>().getElementType() << ">";
}

void printResultType(StringRef name, Type type, DialectAsmPrinter &printer) {
  printer << name;
  printer << "<" << type.cast<ResultType>().getResultType() << ">";
}

} // namespace

void VtcDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<FieldType>(
          [&](Type) { printGridOrTempType(getFieldTypeName(), type, printer); })
      .Case<TempType>(
          [&](Type) { printGridOrTempType(getTempTypeName(), type, printer); })
      .Case<ResultType>(
          [&](Type) { printResultType(getResultTypeName(), type, printer); })
      .Default([](Type) { llvm_unreachable("unexpected 'shape' type kind"); });
}
