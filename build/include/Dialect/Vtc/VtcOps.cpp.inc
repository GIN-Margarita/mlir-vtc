/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::Vtc::AccessOp,
::mlir::Vtc::ApplyOp,
::mlir::Vtc::BufferOp,
::mlir::Vtc::CastOp,
::mlir::Vtc::CombineOp,
::mlir::Vtc::DynAccessOp,
::mlir::Vtc::IndexOp,
::mlir::Vtc::LoadOp,
::mlir::Vtc::ReturnOp,
::mlir::Vtc::StoreOp,
::mlir::Vtc::StoreResultOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace Vtc {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::Vtc::TempType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be a temporary value, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isF32())) || ((type.isF64())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit float or 64-bit float, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::Vtc::TempType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be a temporary value, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::Vtc::FieldType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be a input/output array, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::Vtc::ResultType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be a result, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isF32())) || ((type.isF64())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit float or 64-bit float, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VtcOps9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::Vtc::ResultType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be a result, but got " << type;
  }
  return ::mlir::success();
}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::AccessOp definitions
//===----------------------------------------------------------------------===//

AccessOpAdaptor::AccessOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AccessOpAdaptor::AccessOpAdaptor(AccessOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AccessOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AccessOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AccessOpAdaptor::temp() {
  return *getODSOperands(0).begin();
}

::mlir::ArrayAttr AccessOpAdaptor::offset() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offset").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult AccessOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_offset = odsAttrs.get("offset");
  if (!tblgen_offset) return emitError(loc, "'Vtc.access' op ""requires attribute 'offset'");
    if (!((((tblgen_offset.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_offset.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_offset.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.access' op ""attribute 'offset' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  return ::mlir::success();
}

::llvm::StringRef AccessOp::getOperationName() {
  return "Vtc.access";
}

std::pair<unsigned, unsigned> AccessOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AccessOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AccessOp::temp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AccessOp::tempMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AccessOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AccessOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AccessOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr AccessOp::offsetAttr() {
  return (*this)->getAttr("offset").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AccessOp::offset() {
  auto attr = offsetAttr();
  return attr;
}

void AccessOp::offsetAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("offset", attr);
}

void AccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp, ArrayRef<int64_t> offset) {
            odsState.addOperands(temp);
            odsState.addAttribute(getOffsetAttrName(), odsBuilder.getI64ArrayAttr(offset));
            odsState.addTypes(temp.getType().cast<Vtc::GridType>().getElementType());
        
}

void AccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value temp, ::mlir::ArrayAttr offset) {
  odsState.addOperands(temp);
  odsState.addAttribute("offset", offset);
  odsState.addTypes(res);
}

void AccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, ::mlir::ArrayAttr offset) {
  odsState.addOperands(temp);
  odsState.addAttribute("offset", offset);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AccessOp::verify() {
  if (failed(AccessOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  auto tempType = temp().getType().cast<Vtc::GridType>();

        if(offset().size() != tempType.getRank())
            return emitOpError("expected offset to have the same rank as the temp type");
        if(res().getType() != tempType.getElementType())
            return emitOpError("expected result to have the same element type as the temp type");

        return success();
}

::mlir::ParseResult AccessOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tempRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tempOperands(tempRawOperands);  ::llvm::SMLoc tempOperandsLoc;
  (void)tempOperandsLoc;
  ::mlir::ArrayAttr offsetAttr;
  ::llvm::ArrayRef<::mlir::Type> tempTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  tempOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tempRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(offsetAttr, parser.getBuilder().getType<::mlir::NoneType>(), "offset", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType temp__res_functionType;
  if (parser.parseType(temp__res_functionType))
    return ::mlir::failure();
  tempTypes = temp__res_functionType.getInputs();
  resTypes = temp__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(tempOperands, tempTypes, tempOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AccessOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.access";
  p << ' ';
  p << temp();
  p << "(";
  p.printAttributeWithoutType(offsetAttr());
  p << ")";
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"offset"});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(temp().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

void AccessOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::ApplyOp definitions
//===----------------------------------------------------------------------===//

ApplyOpAdaptor::ApplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ApplyOpAdaptor::ApplyOpAdaptor(ApplyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ApplyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ApplyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ArrayAttr ApplyOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ApplyOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ApplyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (tblgen_lb) {
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.apply' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (tblgen_ub) {
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.apply' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef ApplyOp::getOperationName() {
  return "Vtc.apply";
}

std::pair<unsigned, unsigned> ApplyOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ApplyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ApplyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ApplyOp::res() {
  return getODSResults(0);
}

::mlir::Region &ApplyOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr ApplyOp::lbAttr() {
  return (*this)->getAttr("lb").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyOp::lb() {
  auto attr = lbAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr ApplyOp::ubAttr() {
  return (*this)->getAttr("ub").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyOp::ub() {
  auto attr = ubAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ApplyOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void ApplyOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

::mlir::Attribute ApplyOp::removeLbAttr() {
  return (*this)->removeAttr("lb");
}

::mlir::Attribute ApplyOp::removeUbAttr() {
  return (*this)->removeAttr("ub");
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange operands, Optional<ArrayAttr> lb, Optional<ArrayAttr> ub) {
      odsState.addOperands(operands);
      if(lb.hasValue() && ub.hasValue()) {
        odsState.addAttribute(getLBAttrName(), lb.getValue());
        odsState.addAttribute(getUBAttrName(), ub.getValue()); 
      }
      auto region = odsState.addRegion(); 
      region->push_back(new Block());
      region->front().addArguments(operands.getTypes());
      odsState.addTypes(resultTypes);
    
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Optional<ArrayAttr> lb, Optional<ArrayAttr> ub) {
      if(lb.hasValue() && ub.hasValue()) {
        odsState.addAttribute(getLBAttrName(), lb.getValue());
        odsState.addAttribute(getUBAttrName(), ub.getValue()); 
      }
      odsState.addRegion();
      odsState.addTypes(resultTypes);
    
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange res, ::mlir::ValueRange operands, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(operands);
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  (void)odsState.addRegion();
  odsState.addTypes(res);
}

void ApplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ApplyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseApplyOp(parser, result);
}

void ApplyOp::print(::mlir::OpAsmPrinter &p) {
  ::print(*this, p);
}

::mlir::LogicalResult ApplyOp::verify() {
  if (failed(ApplyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  // Check the operands
    if (region().front().getNumArguments() != operands().size())
      return emitOpError("operand and argument counts do not match");
    for (unsigned i = 0, e = operands().size(); i != e; ++i) {
      if (region().front().getArgument(i).getType() != operands()[i].getType())
        return emitOpError("operand and argument types do not match");
    }

    // Check the results
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    for(auto result : res()) {
      auto tempType = result.getType().cast<GridType>();
      if(shapeOp.hasShape()) {
        if(shapeOp.getRank() != tempType.getRank())
          return emitOpError("expected result rank to match the operation rank");
        if (!tempType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
          return emitOpError("expected temp type to have the shape of the op");
      }
    }
    return success();
}





void ApplyOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::BufferOp definitions
//===----------------------------------------------------------------------===//

BufferOpAdaptor::BufferOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BufferOpAdaptor::BufferOpAdaptor(BufferOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BufferOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BufferOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BufferOpAdaptor::temp() {
  return *getODSOperands(0).begin();
}

::mlir::ArrayAttr BufferOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr BufferOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult BufferOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (tblgen_lb) {
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.buffer' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (tblgen_ub) {
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.buffer' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BufferOp::getOperationName() {
  return "Vtc.buffer";
}

std::pair<unsigned, unsigned> BufferOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BufferOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BufferOp::temp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BufferOp::tempMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BufferOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BufferOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BufferOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr BufferOp::lbAttr() {
  return (*this)->getAttr("lb").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > BufferOp::lb() {
  auto attr = lbAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr BufferOp::ubAttr() {
  return (*this)->getAttr("ub").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > BufferOp::ub() {
  auto attr = ubAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void BufferOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void BufferOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

::mlir::Attribute BufferOp::removeLbAttr() {
  return (*this)->removeAttr("lb");
}

::mlir::Attribute BufferOp::removeUbAttr() {
  return (*this)->removeAttr("ub");
}

void BufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp) {
      odsState.addOperands(temp);
      odsState.addTypes(temp.getType());
    
}

void BufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value temp, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(temp);
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  odsState.addTypes(res);
}

void BufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(temp);
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BufferOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BufferOp::verify() {
  if (failed(BufferOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
  // Check the temp and result types
    auto tempType = temp().getType().cast<Vtc::GridType>();
    auto resType = res().getType().cast<Vtc::GridType>();
    if (resType.getRank() != tempType.getRank())
      return emitOpError("the result and temp types have different rank");
    if (resType.getAllocation() != tempType.getAllocation())
      return emitOpError("the result and temp types have different allocation");
    if (resType.getElementType() != tempType.getElementType())
      return emitOpError("the result and temp types have different element types");

    // Ensure the shape matches the temp and result types
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if(shapeOp.hasShape()) {
      if (!tempType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected the temp type to be larger than the op shape");
      if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected op and result type to have the same shape");
    }

    if(!(isa<Vtc::ApplyOp>(temp().getDefiningOp()) ||
         isa<Vtc::CombineOp>(temp().getDefiningOp())))
      return emitOpError("expected buffer to connect to an apply or combine op");
    
    if (!llvm::all_of(temp().getUsers(), [](Operation *op) { return isa<Vtc::BufferOp>(op); })) 
      return emitOpError("expected only buffers use the same value");
    return success();
}



::mlir::ParseResult BufferOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tempRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tempOperands(tempRawOperands);  ::llvm::SMLoc tempOperandsLoc;
  (void)tempOperandsLoc;
  ::mlir::ArrayAttr lbAttr;
  ::mlir::ArrayAttr ubAttr;
  ::llvm::ArrayRef<::mlir::Type> tempTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  tempOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tempRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(lbAttr, parser.getBuilder().getType<::mlir::NoneType>(), "lb", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(ubAttr, parser.getBuilder().getType<::mlir::NoneType>(), "ub", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType temp__res_functionType;
  if (parser.parseType(temp__res_functionType))
    return ::mlir::failure();
  tempTypes = temp__res_functionType.getInputs();
  resTypes = temp__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(tempOperands, tempTypes, tempOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BufferOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.buffer";
  p << ' ';
  p << temp();
  if ((*this)->getAttr("lb")) {
  p << "(";
  p.printAttributeWithoutType(lbAttr());
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(ubAttr());
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"lb", "ub"});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(temp().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

void BufferOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

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
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps4(getOperation(), v.getType(), "result", index)))
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
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::CombineOp definitions
//===----------------------------------------------------------------------===//

CombineOpAdaptor::CombineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CombineOpAdaptor::CombineOpAdaptor(CombineOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CombineOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange CombineOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CombineOpAdaptor::lower() {
  return getODSOperands(0);
}

::mlir::ValueRange CombineOpAdaptor::upper() {
  return getODSOperands(1);
}

::mlir::ValueRange CombineOpAdaptor::lowerext() {
  return getODSOperands(2);
}

::mlir::ValueRange CombineOpAdaptor::upperext() {
  return getODSOperands(3);
}

::mlir::IntegerAttr CombineOpAdaptor::dim() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("dim").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr CombineOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr CombineOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CombineOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult CombineOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements, but got ") << numElements;
  }
    {
  auto tblgen_dim = odsAttrs.get("dim");
  if (!tblgen_dim) return emitError(loc, "'Vtc.combine' op ""requires attribute 'dim'");
    if (!((((tblgen_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getInt() <= 2)))) return emitError(loc, "'Vtc.combine' op ""attribute 'dim' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0 whose maximum value is 2");
  }
  {
  auto tblgen_index = odsAttrs.get("index");
  if (!tblgen_index) return emitError(loc, "'Vtc.combine' op ""requires attribute 'index'");
    if (!(((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'Vtc.combine' op ""attribute 'index' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (tblgen_lb) {
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.combine' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (tblgen_ub) {
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.combine' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef CombineOp::getOperationName() {
  return "Vtc.combine";
}

std::pair<unsigned, unsigned> CombineOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range CombineOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CombineOp::lower() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range CombineOp::upper() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CombineOp::lowerext() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range CombineOp::upperext() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange CombineOp::lowerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CombineOp::upperMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CombineOp::lowerextMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CombineOp::upperextMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> CombineOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range CombineOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CombineOp::res() {
  return getODSResults(0);
}

::mlir::IntegerAttr CombineOp::dimAttr() {
  return (*this)->getAttr("dim").template cast<::mlir::IntegerAttr>();
}

uint64_t CombineOp::dim() {
  auto attr = dimAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CombineOp::indexAttr() {
  return (*this)->getAttr("index").template cast<::mlir::IntegerAttr>();
}

uint64_t CombineOp::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr CombineOp::lbAttr() {
  return (*this)->getAttr("lb").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CombineOp::lb() {
  auto attr = lbAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CombineOp::ubAttr() {
  return (*this)->getAttr("ub").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CombineOp::ub() {
  auto attr = ubAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void CombineOp::dimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("dim", attr);
}

void CombineOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

void CombineOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void CombineOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

::mlir::Attribute CombineOp::removeLbAttr() {
  return (*this)->removeAttr("lb");
}

::mlir::Attribute CombineOp::removeUbAttr() {
  return (*this)->removeAttr("ub");
}

void CombineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange res, ::mlir::IntegerAttr dim, ::mlir::IntegerAttr index, ::mlir::ValueRange lower, ::mlir::ValueRange upper, ::mlir::ValueRange lowerext, ::mlir::ValueRange upperext, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(lower);
  odsState.addOperands(upper);
  odsState.addOperands(lowerext);
  odsState.addOperands(upperext);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(lower.size()), static_cast<int32_t>(upper.size()), static_cast<int32_t>(lowerext.size()), static_cast<int32_t>(upperext.size())}));
  odsState.addAttribute("dim", dim);
  odsState.addAttribute("index", index);
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  odsState.addTypes(res);
}

void CombineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange res, uint64_t dim, uint64_t index, ::mlir::ValueRange lower, ::mlir::ValueRange upper, ::mlir::ValueRange lowerext, ::mlir::ValueRange upperext, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(lower);
  odsState.addOperands(upper);
  odsState.addOperands(lowerext);
  odsState.addOperands(upperext);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(lower.size()), static_cast<int32_t>(upper.size()), static_cast<int32_t>(lowerext.size()), static_cast<int32_t>(upperext.size())}));
  odsState.addAttribute("dim", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dim));
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), index));
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  odsState.addTypes(res);
}

void CombineOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CombineOp::verify() {
  if (failed(CombineOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}





::mlir::ParseResult CombineOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr dimAttr;
  ::mlir::IntegerAttr indexAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> lowerOperands;
  ::llvm::SMLoc lowerOperandsLoc;
  (void)lowerOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> lowerTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> upperOperands;
  ::llvm::SMLoc upperOperandsLoc;
  (void)upperOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> upperTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> lowerextOperands;
  ::llvm::SMLoc lowerextOperandsLoc;
  (void)lowerextOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> lowerextTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> upperextOperands;
  ::llvm::SMLoc upperextOperandsLoc;
  (void)upperextOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> upperextTypes;
  ::mlir::ArrayAttr lbAttr;
  ::mlir::ArrayAttr ubAttr;
  ::mlir::SmallVector<::mlir::Type, 1> resTypes;

  if (parser.parseAttribute(dimAttr, parser.getBuilder().getIntegerType(64), "dim", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("at"))
    return ::mlir::failure();

  if (parser.parseAttribute(indexAttr, parser.getBuilder().getIntegerType(64), "index", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("lower"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  lowerOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(lowerOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(lowerTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseKeyword("upper"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  upperOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(upperOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(upperTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("lowerext"))) {
  if (parser.parseEqual())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  lowerextOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(lowerextOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(lowerextTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("upperext"))) {
  if (parser.parseEqual())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  upperextOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(upperextOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(upperextTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalLParen())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(lbAttr, parser.getBuilder().getType<::mlir::NoneType>(), "lb", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(ubAttr, parser.getBuilder().getType<::mlir::NoneType>(), "ub", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resTypes))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lowerOperands, lowerTypes, lowerOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperOperands, upperTypes, upperOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(lowerextOperands, lowerextTypes, lowerextOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperextOperands, upperextTypes, upperextOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(lowerOperands.size()), static_cast<int32_t>(upperOperands.size()), static_cast<int32_t>(lowerextOperands.size()), static_cast<int32_t>(upperextOperands.size())}));
  return ::mlir::success();
}

void CombineOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.combine";
  p << ' ';
  p.printAttributeWithoutType(dimAttr());
  p << ' ' << "at";
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  p << ' ' << "lower";
  p << ' ' << "=";
  p << ' ' << "(";
  p << lower();
  p << ' ' << ":";
  p << ' ';
  p << lower().getTypes();
  p << ")";
  p << ' ' << "upper";
  p << ' ' << "=";
  p << ' ' << "(";
  p << upper();
  p << ' ' << ":";
  p << ' ';
  p << upper().getTypes();
  p << ")";
  if (!lowerext().empty()) {
  p << ' ' << "lowerext";
  p << ' ' << "=";
  p << ' ' << "(";
  p << lowerext();
  p << ' ' << ":";
  p << ' ';
  p << lowerext().getTypes();
  p << ")";
  }
  if (!upperext().empty()) {
  p << ' ' << "upperext";
  p << ' ' << "=";
  p << ' ' << "(";
  p << upperext();
  p << ' ' << ":";
  p << ' ';
  p << upperext().getTypes();
  p << ")";
  }
  if ((*this)->getAttr("lb")) {
  p << ' ' << "(";
  p.printAttributeWithoutType(lbAttr());
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(ubAttr());
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "dim", "index", "lb", "ub"});
  p << ' ' << ":";
  p << ' ';
  p << res().getTypes();
}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::DynAccessOp definitions
//===----------------------------------------------------------------------===//

DynAccessOpAdaptor::DynAccessOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

DynAccessOpAdaptor::DynAccessOpAdaptor(DynAccessOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> DynAccessOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange DynAccessOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DynAccessOpAdaptor::temp() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange DynAccessOpAdaptor::offset() {
  return getODSOperands(1);
}

::mlir::ArrayAttr DynAccessOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DynAccessOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult DynAccessOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (!tblgen_lb) return emitError(loc, "'Vtc.dyn_access' op ""requires attribute 'lb'");
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.dyn_access' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (!tblgen_ub) return emitError(loc, "'Vtc.dyn_access' op ""requires attribute 'ub'");
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.dyn_access' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  return ::mlir::success();
}

::llvm::StringRef DynAccessOp::getOperationName() {
  return "Vtc.dyn_access";
}

std::pair<unsigned, unsigned> DynAccessOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range DynAccessOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DynAccessOp::temp() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range DynAccessOp::offset() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DynAccessOp::tempMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange DynAccessOp::offsetMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DynAccessOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DynAccessOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DynAccessOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr DynAccessOp::lbAttr() {
  return (*this)->getAttr("lb").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DynAccessOp::lb() {
  auto attr = lbAttr();
  return attr;
}

::mlir::ArrayAttr DynAccessOp::ubAttr() {
  return (*this)->getAttr("ub").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DynAccessOp::ub() {
  auto attr = ubAttr();
  return attr;
}

void DynAccessOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void DynAccessOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

void DynAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp, ValueRange offset, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) {
      odsState.addOperands(temp);
      odsState.addOperands(offset);
      odsState.addAttribute(getLBAttrName(), odsBuilder.getI64ArrayAttr(lb));
      odsState.addAttribute(getUBAttrName(), odsBuilder.getI64ArrayAttr(ub));
      auto tempType = temp.getType().cast<Vtc::GridType>();
      odsState.addTypes(tempType.getElementType());
    
}

void DynAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value temp, ::mlir::ValueRange offset, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub) {
  odsState.addOperands(temp);
  odsState.addOperands(offset);
  odsState.addAttribute("lb", lb);
  odsState.addAttribute("ub", ub);
  odsState.addTypes(res);
}

void DynAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, ::mlir::ValueRange offset, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub) {
  odsState.addOperands(temp);
  odsState.addOperands(offset);
  odsState.addAttribute("lb", lb);
  odsState.addAttribute("ub", ub);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DynAccessOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DynAccessOp::verify() {
  if (failed(DynAccessOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  auto tempType = temp().getType().cast<Vtc::GridType>();
    if (offset().size() != tempType.getRank())
      return emitOpError("offset and temp dimensions do not match");
    if (res().getType() != tempType.getElementType())
      return emitOpError("result type and element type are inconsistent");
    return success();
}





::mlir::ParseResult DynAccessOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tempRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tempOperands(tempRawOperands);  ::llvm::SMLoc tempOperandsLoc;
  (void)tempOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> offsetOperands;
  ::llvm::SMLoc offsetOperandsLoc;
  (void)offsetOperandsLoc;
  ::mlir::ArrayAttr lbAttr;
  ::mlir::ArrayAttr ubAttr;
  ::llvm::ArrayRef<::mlir::Type> tempTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  tempOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tempRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  offsetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(offsetOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseKeyword("in"))
    return ::mlir::failure();

  if (parser.parseAttribute(lbAttr, parser.getBuilder().getType<::mlir::NoneType>(), "lb", result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseAttribute(ubAttr, parser.getBuilder().getType<::mlir::NoneType>(), "ub", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType temp__res_functionType;
  if (parser.parseType(temp__res_functionType))
    return ::mlir::failure();
  tempTypes = temp__res_functionType.getInputs();
  resTypes = temp__res_functionType.getResults();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resTypes);
  if (parser.resolveOperands(tempOperands, tempTypes, tempOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DynAccessOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.dyn_access";
  p << ' ';
  p << temp();
  p << "(";
  p << offset();
  p << ")";
  p << ' ' << "in";
  p << ' ';
  p.printAttributeWithoutType(lbAttr());
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(ubAttr());
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"lb", "ub"});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(temp().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

void DynAccessOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::IndexOp definitions
//===----------------------------------------------------------------------===//

IndexOpAdaptor::IndexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

IndexOpAdaptor::IndexOpAdaptor(IndexOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> IndexOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IndexOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr IndexOpAdaptor::dim() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("dim").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr IndexOpAdaptor::offset() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offset").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult IndexOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dim = odsAttrs.get("dim");
  if (!tblgen_dim) return emitError(loc, "'Vtc.index' op ""requires attribute 'dim'");
    if (!((((tblgen_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getInt() <= 2)))) return emitError(loc, "'Vtc.index' op ""attribute 'dim' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0 whose maximum value is 2");
  }
  {
  auto tblgen_offset = odsAttrs.get("offset");
  if (!tblgen_offset) return emitError(loc, "'Vtc.index' op ""requires attribute 'offset'");
    if (!((((tblgen_offset.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_offset.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_offset.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.index' op ""attribute 'offset' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  return ::mlir::success();
}

::llvm::StringRef IndexOp::getOperationName() {
  return "Vtc.index";
}

std::pair<unsigned, unsigned> IndexOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IndexOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> IndexOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IndexOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IndexOp::idx() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr IndexOp::dimAttr() {
  return (*this)->getAttr("dim").template cast<::mlir::IntegerAttr>();
}

uint64_t IndexOp::dim() {
  auto attr = dimAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr IndexOp::offsetAttr() {
  return (*this)->getAttr("offset").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr IndexOp::offset() {
  auto attr = offsetAttr();
  return attr;
}

void IndexOp::dimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("dim", attr);
}

void IndexOp::offsetAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("offset", attr);
}

void IndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int64_t dim, ArrayRef<int64_t> offset) {
            odsState.addAttribute(getDimAttrName(), odsBuilder.getI64IntegerAttr(dim));
            odsState.addAttribute(getOffsetAttrName(), odsBuilder.getI64ArrayAttr(offset));
            odsState.addTypes(odsBuilder.getIndexType());
        
}

void IndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type idx, ::mlir::IntegerAttr dim, ::mlir::ArrayAttr offset) {
  odsState.addAttribute("dim", dim);
  odsState.addAttribute("offset", offset);
  odsState.addTypes(idx);
}

void IndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dim, ::mlir::ArrayAttr offset) {
  odsState.addAttribute("dim", dim);
  odsState.addAttribute("offset", offset);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type idx, uint64_t dim, ::mlir::ArrayAttr offset) {
  odsState.addAttribute("dim", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dim));
  odsState.addAttribute("offset", offset);
  odsState.addTypes(idx);
}

void IndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t dim, ::mlir::ArrayAttr offset) {
  odsState.addAttribute("dim", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dim));
  odsState.addAttribute("offset", offset);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IndexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IndexOp::verify() {
  if (failed(IndexOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps6(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return success();
}

::mlir::ParseResult IndexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr dimAttr;
  ::mlir::ArrayAttr offsetAttr;
  ::mlir::Type idxRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> idxTypes(idxRawTypes);

  if (parser.parseAttribute(dimAttr, parser.getBuilder().getIntegerType(64), "dim", result.attributes))
    return ::mlir::failure();

  if (parser.parseAttribute(offsetAttr, parser.getBuilder().getType<::mlir::NoneType>(), "offset", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(idxRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(idxTypes);
  return ::mlir::success();
}

void IndexOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.index";
  p << ' ';
  p.printAttributeWithoutType(dimAttr());
  p << ' ';
  p.printAttributeWithoutType(offsetAttr());
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"dim", "offset"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(idx().getType());
}

void IndexOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::LoadOp definitions
//===----------------------------------------------------------------------===//

LoadOpAdaptor::LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

LoadOpAdaptor::LoadOpAdaptor(LoadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> LoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOpAdaptor::field() {
  return *getODSOperands(0).begin();
}

::mlir::ArrayAttr LoadOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr LoadOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (tblgen_lb) {
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.load' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (tblgen_ub) {
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.load' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef LoadOp::getOperationName() {
  return "Vtc.load";
}

std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::field() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LoadOp::fieldMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr LoadOp::lbAttr() {
  return (*this)->getAttr("lb").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOp::lb() {
  auto attr = lbAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr LoadOp::ubAttr() {
  return (*this)->getAttr("ub").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOp::ub() {
  auto attr = ubAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void LoadOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void LoadOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

::mlir::Attribute LoadOp::removeLbAttr() {
  return (*this)->removeAttr("lb");
}

::mlir::Attribute LoadOp::removeUbAttr() {
  return (*this)->removeAttr("ub");
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value field, Optional<ArrayAttr> lb, Optional<ArrayAttr> ub) {
      odsState.addOperands(field);
      auto fieldType = field.getType().cast<Vtc::GridType>();
      if(lb.hasValue() && ub.hasValue()) {
        odsState.addAttribute(getLBAttrName(), lb.getValue());
        odsState.addAttribute(getUBAttrName(), ub.getValue()); 
        Index lower, upper;
        for (auto &elem : lb.getValue()) 
          lower.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        for (auto &elem : ub.getValue()) 
          upper.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        odsState.addTypes(Vtc::TempType::get(fieldType.getElementType(), fieldType.getAllocation(), lower, upper));
      } else {
        odsState.addTypes(Vtc::TempType::get(fieldType.getElementType(), fieldType.getShape()));
      }
    
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value field, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(field);
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value field, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub) {
  odsState.addOperands(field);
  if (lb) {
  odsState.addAttribute("lb", lb);
  }
  if (ub) {
  odsState.addAttribute("ub", ub);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoadOp::verify() {
  if (failed(LoadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps4(getOperation(), v.getType(), "operand", index)))
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
  // Check the field and result types
    auto fieldType = field().getType().cast<Vtc::GridType>();
    auto resType = res().getType().cast<Vtc::GridType>();
    if(fieldType.hasDynamicShape())
      return emitOpError("expected fields to have static shape");
    if (fieldType.getRank() != resType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and temp types have different element types");
    
    // Ensure the shape matches the field and result types
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if(shapeOp.hasShape()) {
      if (!fieldType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected the field type to be larger than the op shape");
      if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected op and result type to have the same shape");
    }

    if(!isa<Vtc::CastOp>(field().getDefiningOp()))
      return emitOpError("expected the defining op of the field is a cast operation");

    return success();
}





::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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
  if (succeeded(parser.parseOptionalLParen())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(lbAttr, parser.getBuilder().getType<::mlir::NoneType>(), "lb", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(ubAttr, parser.getBuilder().getType<::mlir::NoneType>(), "ub", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
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

void LoadOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.load";
  p << ' ';
  p << field();
  if ((*this)->getAttr("lb")) {
  p << "(";
  p.printAttributeWithoutType(lbAttr());
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(ubAttr());
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"lb", "ub"});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(field().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

void LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ArrayAttr ReturnOpAdaptor::unroll() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("unroll").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_unroll = odsAttrs.get("unroll");
  if (tblgen_unroll) {
    if (!((((tblgen_unroll.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_unroll.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_unroll.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.return' op ""attribute 'unroll' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef ReturnOp::getOperationName() {
  return "Vtc.return";
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ReturnOp::unrollAttr() {
  return (*this)->getAttr("unroll").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ReturnOp::unroll() {
  auto attr = unrollAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ReturnOp::unrollAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("unroll", attr);
}

::mlir::Attribute ReturnOp::removeUnrollAttr() {
  return (*this)->removeAttr("unroll");
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, Optional<ArrayAttr> unroll) {
      odsState.addOperands({operands});
      if(unroll.hasValue())
        odsState.addAttribute(getUnrollAttrName(), unroll.getValue());
    
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, /*optional*/::mlir::ArrayAttr unroll) {
  odsState.addOperands(operands);
  if (unroll) {
  odsState.addAttribute("unroll", unroll);
  }
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/::mlir::ArrayAttr unroll) {
  odsState.addOperands(operands);
  if (unroll) {
  odsState.addAttribute("unroll", unroll);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (failed(ReturnOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  auto applyOp = cast<Vtc::ApplyOp>(getOperation()->getParentOp());
    unsigned unrollFac = getUnrollFac();

    // Verify the number of operands matches the number of apply results
    auto results = applyOp.res();
    if (getNumOperands() != unrollFac * results.size())
      return emitOpError("the operand and apply result counts do not match");
  
    // Verify the element types match
    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      auto tempType = applyOp.getResult(i).getType().cast<GridType>();
      for (unsigned j = 0; j < unrollFac; j++)
        if (getOperand(i * unrollFac + j).getType().cast<Vtc::ResultType>().getResultType() != tempType.getElementType())
          return emitOpError("the operand and apply result element types do not match");
    }
    return success();
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr unrollAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
  if (succeeded(parser.parseOptionalKeyword("unroll"))) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(unrollAttr, parser.getBuilder().getType<::mlir::NoneType>(), "unroll", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  }

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.return";
  if ((*this)->getAttr("unroll")) {
  p << ' ' << "unroll";
  p << ' ';
  p.printAttributeWithoutType(unrollAttr());
  }
  p << ' ';
  p << operands();
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"unroll"});
  p << ' ' << ":";
  p << ' ';
  p << operands().getTypes();
}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::StoreOp definitions
//===----------------------------------------------------------------------===//

StoreOpAdaptor::StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

StoreOpAdaptor::StoreOpAdaptor(StoreOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> StoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOpAdaptor::temp() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOpAdaptor::field() {
  return *getODSOperands(1).begin();
}

::mlir::ArrayAttr StoreOpAdaptor::lb() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("lb").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr StoreOpAdaptor::ub() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("ub").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lb = odsAttrs.get("lb");
  if (!tblgen_lb) return emitError(loc, "'Vtc.store' op ""requires attribute 'lb'");
    if (!((((tblgen_lb.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_lb.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_lb.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.store' op ""attribute 'lb' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  {
  auto tblgen_ub = odsAttrs.get("ub");
  if (!tblgen_ub) return emitError(loc, "'Vtc.store' op ""requires attribute 'ub'");
    if (!((((tblgen_ub.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ub.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_ub.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'Vtc.store' op ""attribute 'ub' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  return ::mlir::success();
}

::llvm::StringRef StoreOp::getOperationName() {
  return "Vtc.store";
}

std::pair<unsigned, unsigned> StoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOp::temp() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOp::field() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange StoreOp::tempMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange StoreOp::fieldMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr StoreOp::lbAttr() {
  return (*this)->getAttr("lb").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr StoreOp::lb() {
  auto attr = lbAttr();
  return attr;
}

::mlir::ArrayAttr StoreOp::ubAttr() {
  return (*this)->getAttr("ub").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr StoreOp::ub() {
  auto attr = ubAttr();
  return attr;
}

void StoreOp::lbAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("lb", attr);
}

void StoreOp::ubAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("ub", attr);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp, Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) {
      odsState.addOperands({temp, field});
      odsState.addAttribute(getLBAttrName(), odsBuilder.getI64ArrayAttr(lb));
      odsState.addAttribute(getUBAttrName(), odsBuilder.getI64ArrayAttr(ub)); 
    
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value temp, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub) {
  odsState.addOperands(temp);
  odsState.addOperands(field);
  odsState.addAttribute("lb", lb);
  odsState.addAttribute("ub", ub);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub) {
  odsState.addOperands(temp);
  odsState.addOperands(field);
  odsState.addAttribute("lb", lb);
  odsState.addAttribute("ub", ub);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StoreOp::verify() {
  if (failed(StoreOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  // Check the field and result types
    auto fieldType = field().getType().cast<Vtc::GridType>();
    auto tempType = temp().getType().cast<Vtc::GridType>();
    if(fieldType.hasDynamicShape())
      return emitOpError("expected fields to have static shape");
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != tempType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != tempType.getElementType())
      return emitOpError("the field and temp types have different element types");

    // Ensure the shape matches the temp and result types
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if (!fieldType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected the field type to be larger than the op shape");
    if (!tempType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected the temp type to be larger than the op shape");

    if (!(dyn_cast<Vtc::ApplyOp>(temp().getDefiningOp()) || 
          dyn_cast<Vtc::CombineOp>(temp().getDefiningOp())))
      return emitOpError("output temp not result of an apply or a combine op");
    if (llvm::count_if(field().getUsers(), [](Operation *op) { return isa_and_nonnull<Vtc::LoadOp>(op); }) != 0)
      return emitOpError("an output cannot be an input");  
    if (llvm::count_if(field().getUsers(), [](Operation *op) { return isa_and_nonnull<Vtc::StoreOp>(op); }) != 1)
      return emitOpError("multiple stores to the same output");  
    
    if(!isa<Vtc::CastOp>(field().getDefiningOp()))
      return emitOpError("expected the defining op of the field is a cast operation");

    return success();
}





::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tempRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tempOperands(tempRawOperands);  ::llvm::SMLoc tempOperandsLoc;
  (void)tempOperandsLoc;
  ::mlir::OpAsmParser::OperandType fieldRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> fieldOperands(fieldRawOperands);  ::llvm::SMLoc fieldOperandsLoc;
  (void)fieldOperandsLoc;
  ::mlir::ArrayAttr lbAttr;
  ::mlir::ArrayAttr ubAttr;
  ::mlir::Type tempRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tempTypes(tempRawTypes);
  ::mlir::Type fieldRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> fieldTypes(fieldRawTypes);

  tempOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tempRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

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

  if (parser.parseType(tempRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(fieldRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(tempOperands, tempTypes, tempOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(fieldOperands, fieldTypes, fieldOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.store";
  p << ' ';
  p << temp();
  p << ' ' << "to";
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
  p << ::llvm::ArrayRef<::mlir::Type>(temp().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(field().getType());
}

} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::StoreResultOp definitions
//===----------------------------------------------------------------------===//

StoreResultOpAdaptor::StoreResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

StoreResultOpAdaptor::StoreResultOpAdaptor(StoreResultOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> StoreResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StoreResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange StoreResultOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::LogicalResult StoreResultOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef StoreResultOp::getOperationName() {
  return "Vtc.store_result";
}

std::pair<unsigned, unsigned> StoreResultOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StoreResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range StoreResultOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange StoreResultOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StoreResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreResultOp::res() {
  return *getODSResults(0).begin();
}

void StoreResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      odsState.addOperands({value});
      odsState.addTypes(Vtc::ResultType::get(value.getType()));
    
}

void StoreResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addTypes(res);
}

void StoreResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StoreResultOp::verify() {
  if (failed(StoreResultOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VtcOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  // Check at most one operand
    if(operands().size() > 1)
      return emitOpError("expected at most one operand");

    // Check the return type
    auto resultType = res().getType().cast<Vtc::ResultType>().getResultType();
    if(operands().size() == 1 && resultType != operands()[0].getType())
      return emitOpError("operand type and result type are inconsistent");
      
    // Check the result mapping
    if(!getReturnOpOperands())
      return emitOpError("expected valid mapping to return op operands");
    return success();
}

::mlir::ParseResult StoreResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> operandsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType operands__res_functionType;
  if (parser.parseType(operands__res_functionType))
    return ::mlir::failure();
  operandsTypes = operands__res_functionType.getInputs();
  resTypes = operands__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreResultOp::print(::mlir::OpAsmPrinter &p) {
  p << "Vtc.store_result";
  if (!operands().empty()) {
  p << ' ';
  p << operands();
  }
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(operands().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

void StoreResultOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace Vtc
} // namespace mlir

#endif  // GET_OP_CLASSES

