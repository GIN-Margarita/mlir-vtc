/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::Vtc::CastOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace Vtc {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::Vtc::FieldType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be a input/output array, but got " << type;
  }
  return ::mlir::success();
}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CastOpAdaptor::CastOpAdaptor(CastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::field() {
  return *getODSOperands(0).begin();
}

::mlir::ArrayAttr CastOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CastOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (!tblgen_lb) return emitError(loc, "'Vtc.cast' op ""requires attribute 'lb'");
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.cast' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (!tblgen_ub) return emitError(loc, "'Vtc.cast' op ""requires attribute 'ub'");
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.cast' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  return ::mlir::success();
}

::llvm::StringRef CastOp::getOperationName() {
  return "Vtc.cast";
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::field() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::fieldMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr CastOp::lbAttr() {
  return (*this)->getAttr("lb").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CastOp::lb() {
  auto attr = lbAttr();
  return attr;
}

::mlir::ArrayAttr CastOp::ubAttr() {
  return (*this)->getAttr("ub").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CastOp::ub() {
  auto attr = ubAttr();
  return attr;
}

void CastOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void CastOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) {
      odsState.addOperands(field);
      odsState.addAttribute(getLBAttrName(), odsBuilder.getI64ArrayAttr(lb));
      odsState.addAttribute(getUBAttrName(), odsBuilder.getI64ArrayAttr(ub));
      odsState.addTypes(Vtc::FieldType::get(
        field.getType().cast<Vtc::GridType>().getElementType(), 
        applyFunElementWise(lb, ub, std::minus<int64_t>())));
    
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub) {
  odsState.addOperands(field);
  odsState.addAttribute("lb", lb);
  odsState.addAttribute("ub", ub);
  odsState.addTypes(res);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub) {
  odsState.addOperands(field);
  odsState.addAttribute("lb", lb);
  odsState.addAttribute("ub", ub);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verify() {
  if (failed(CastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  auto fieldType = field().getType().cast<Vtc::GridType>();
    auto resType = res().getType().cast<Vtc::GridType>();
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if(!fieldType.hasDynamicShape())
      return emitOpError("expected field to have dynamic shape");
    if(resType.hasDynamicShape())
      return emitOpError("expected result to have static shape");
    if(fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("expected the field and result types to have the same allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and result types have different element types");
    if(shapeOp.getRank() != fieldType.getRank() || shapeOp.getRank() != resType.getRank())
      return emitOpError("expected op and the field and result types to have the same rank");
    
    // Ensure the shape matches the result type
    if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected op and result type to have the same shape");

    // Verify all users fit the shape
    for(auto user : res().getUsers()) {
      if(auto userOp = dyn_cast<ShapeInterface>(user)) {
        if(userOp.hasShape() &&
           (shapeOp.getLB() != applyFunElementWise(shapeOp.getLB(), userOp.getLB(), min) ||
            shapeOp.getUB() != applyFunElementWise(shapeOp.getUB(), userOp.getUB(), max))) 
          return emitOpError("shape not large enough to fit all accesses");
      }
    }

    return success();
}





::mlir::ParseResult CastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType fieldRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> fieldOperands(fieldRawOperands);  ::llvm::SMLoc fieldOperandsLoc;
  (void)fieldOperandsLoc;
  ::mlir::ArrayAttr lbAttr;
  ::mlir::ArrayAttr ubAttr;
  ::llvm::ArrayRef<::mlir::Type> fieldTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  fieldOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(fieldRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(lbAttr, parser.getBuilder().getType<::mlir::NoneType>(), "lb", result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseAttribute(ubAttr, parser.getBuilder().getType<::mlir::NoneType>(), "ub", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType field__res_functionType;
  if (parser.parseType(field__res_functionType))
    return ::mlir::failure();
  fieldTypes = field__res_functionType.getInputs();
  resTypes = field__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(fieldOperands, fieldTypes, fieldOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CastOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.cast";
  p << ' ';
  p << field();
  p << "(";
  p.printAttributeWithoutType(lbAttr());
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(ubAttr());
  p << ")";
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"lb", "ub"});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(field().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace Vtc
} // namespace mlir

#endif  // GET_OP_CLASSES

