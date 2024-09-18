/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace Vtc {
class AccessOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class ApplyOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class BufferOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class CastOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class CombineOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class DynAccessOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class IndexOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class LoadOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class ReturnOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class StoreOp;
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {
class StoreResultOp;
} // namespace Vtc
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::AccessOp declarations
//===----------------------------------------------------------------------===//

class AccessOpAdaptor {
public:
  AccessOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  AccessOpAdaptor(AccessOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::ArrayAttr offset();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class AccessOp : public ::mlir::Op<AccessOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, OffsetInterface::Trait, ShiftInterface::Trait, ExtentInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AccessOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::MutableOperandRange tempMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr offsetAttr();
  ::mlir::ArrayAttr offset();
  void offsetAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp, ArrayRef<int64_t> offset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value temp, ::mlir::ArrayAttr offset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, ::mlir::ArrayAttr offset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

        static StringRef getOffsetAttrName() { return "offset"; }
    
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::ApplyOp declarations
//===----------------------------------------------------------------------===//

class ApplyOpAdaptor {
public:
  ApplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  ApplyOpAdaptor(ApplyOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ApplyOp : public ::mlir::Op<ApplyOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ShapeInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ApplyOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range res();
  ::mlir::Region &region();
  ::mlir::ArrayAttr lbAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > lb();
  ::mlir::ArrayAttr ubAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > ub();
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeLbAttr();
  ::mlir::Attribute removeUbAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange operands, Optional<ArrayAttr> lb, Optional<ArrayAttr> ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Optional<ArrayAttr> lb, Optional<ArrayAttr> ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange res, ::mlir::ValueRange operands, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  void updateArgumentTypes();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    bool hasOnlyEmptyStores();
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Block *getBody() { return &region().front(); }
    ShapeInterface getCombineTreeRootShape();
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::BufferOp declarations
//===----------------------------------------------------------------------===//

class BufferOpAdaptor {
public:
  BufferOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  BufferOpAdaptor(BufferOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class BufferOp : public ::mlir::Op<BufferOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ShapeInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BufferOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::MutableOperandRange tempMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr lbAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > lb();
  ::mlir::ArrayAttr ubAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > ub();
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeLbAttr();
  ::mlir::Attribute removeUbAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value temp, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void updateArgumentTypes();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::CastOp declarations
//===----------------------------------------------------------------------===//

class CastOpAdaptor {
public:
  CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  CastOpAdaptor(CastOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value field();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class CastOp : public ::mlir::Op<CastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ShapeInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CastOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value field();
  ::mlir::MutableOperandRange fieldMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr lbAttr();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ubAttr();
  ::mlir::ArrayAttr ub();
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  void updateArgumentTypes();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::CombineOp declarations
//===----------------------------------------------------------------------===//

class CombineOpAdaptor {
public:
  CombineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  CombineOpAdaptor(CombineOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange lower();
  ::mlir::ValueRange upper();
  ::mlir::ValueRange lowerext();
  ::mlir::ValueRange upperext();
  ::mlir::IntegerAttr dim();
  ::mlir::IntegerAttr index();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class CombineOp : public ::mlir::Op<CombineOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ShapeInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CombineOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range lower();
  ::mlir::Operation::operand_range upper();
  ::mlir::Operation::operand_range lowerext();
  ::mlir::Operation::operand_range upperext();
  ::mlir::MutableOperandRange lowerMutable();
  ::mlir::MutableOperandRange upperMutable();
  ::mlir::MutableOperandRange lowerextMutable();
  ::mlir::MutableOperandRange upperextMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range res();
  ::mlir::IntegerAttr dimAttr();
  uint64_t dim();
  ::mlir::IntegerAttr indexAttr();
  uint64_t index();
  ::mlir::ArrayAttr lbAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > lb();
  ::mlir::ArrayAttr ubAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > ub();
  void dimAttr(::mlir::IntegerAttr attr);
  void indexAttr(::mlir::IntegerAttr attr);
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeLbAttr();
  ::mlir::Attribute removeUbAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange res, ::mlir::IntegerAttr dim, ::mlir::IntegerAttr index, ::mlir::ValueRange lower, ::mlir::ValueRange upper, ::mlir::ValueRange lowerext, ::mlir::ValueRange upperext, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange res, uint64_t dim, uint64_t index, ::mlir::ValueRange lower, ::mlir::ValueRange upper, ::mlir::ValueRange lowerext, ::mlir::ValueRange upperext, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  void updateArgumentTypes();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    int64_t getIndex() { return indexAttr().getValue().getSExtValue(); }
    SmallVector<Operation*, 4> getLowerDefiningOps() {
      SmallVector<Operation*, 4> definingOps;
      for(auto value : lower()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      for(auto value : lowerext()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      return definingOps;  
    }
    SmallVector<Operation*, 4> getUpperDefiningOps() {
      SmallVector<Operation*, 4> definingOps;
      for(auto value : upper()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      for(auto value : upperext()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      return definingOps;  
    }
    Optional<unsigned> getLowerOperandNumber(unsigned resultNumber) {
      if(resultNumber < lower().size()) {
        return resultNumber;
      }
      return llvm::None;
    }
    Optional<unsigned> getLowerExtraOperandNumber(unsigned resultNumber) {
      if(resultNumber >= lower().size() && resultNumber < lower().size() + lowerext().size()) {
        return resultNumber - lower().size();
      }
      return llvm::None;
    }
    Optional<unsigned> getUpperExtraOperandNumber(unsigned resultNumber) {
      if(resultNumber >= lower().size() + lowerext().size() && resultNumber < lower().size() + lowerext().size() + upperext().size()) {
        return resultNumber - lower().size() - lowerext().size();
      }
      return llvm::None;
    }
    bool isLowerOperand(unsigned operandNumber) {
      return operandNumber < lower().size() ||
        (operandNumber >= lower().size() + upper().size() && 
         operandNumber < lower().size() + upper().size() + lowerext().size());
    }
    bool isUpperOperand(unsigned operandNumber) {
      return !isLowerOperand(operandNumber) && operandNumber < getNumOperands();
    }
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Vtc::CombineOp getCombineTreeRoot();
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::DynAccessOp declarations
//===----------------------------------------------------------------------===//

class DynAccessOpAdaptor {
public:
  DynAccessOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  DynAccessOpAdaptor(DynAccessOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::ValueRange offset();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class DynAccessOp : public ::mlir::Op<DynAccessOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ShiftInterface::Trait, ExtentInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DynAccessOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::Operation::operand_range offset();
  ::mlir::MutableOperandRange tempMutable();
  ::mlir::MutableOperandRange offsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr lbAttr();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ubAttr();
  ::mlir::ArrayAttr ub();
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp, ValueRange offset, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value temp, ::mlir::ValueRange offset, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, ::mlir::ValueRange offset, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void shiftByOffset(mlir::ArrayRef<int64_t> offset);
  std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> getAccessExtent();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::IndexOp declarations
//===----------------------------------------------------------------------===//

class IndexOpAdaptor {
public:
  IndexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  IndexOpAdaptor(IndexOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::IntegerAttr dim();
  ::mlir::ArrayAttr offset();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class IndexOp : public ::mlir::Op<IndexOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, OffsetInterface::Trait, ShiftInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IndexOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value idx();
  ::mlir::IntegerAttr dimAttr();
  uint64_t dim();
  ::mlir::ArrayAttr offsetAttr();
  ::mlir::ArrayAttr offset();
  void dimAttr(::mlir::IntegerAttr attr);
  void offsetAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int64_t dim, ArrayRef<int64_t> offset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type idx, ::mlir::IntegerAttr dim, ::mlir::ArrayAttr offset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dim, ::mlir::ArrayAttr offset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type idx, uint64_t dim, ::mlir::ArrayAttr offset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t dim, ::mlir::ArrayAttr offset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

        static StringRef getDimAttrName() { return "dim"; }
        static StringRef getOffsetAttrName() { return "offset"; }
    
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::LoadOp declarations
//===----------------------------------------------------------------------===//

class LoadOpAdaptor {
public:
  LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  LoadOpAdaptor(LoadOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value field();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class LoadOp : public ::mlir::Op<LoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ShapeInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value field();
  ::mlir::MutableOperandRange fieldMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr lbAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > lb();
  ::mlir::ArrayAttr ubAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > ub();
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeLbAttr();
  ::mlir::Attribute removeUbAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value field, Optional<ArrayAttr> lb, Optional<ArrayAttr> ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value field, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value field, /*optional*/::mlir::ArrayAttr lb, /*optional*/::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  void updateArgumentTypes();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpAdaptor {
public:
  ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  ReturnOpAdaptor(ReturnOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::ArrayAttr unroll();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ReturnOp : public ::mlir::Op<ReturnOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::HasParent<ApplyOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr unrollAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > unroll();
  void unrollAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeUnrollAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, Optional<ArrayAttr> unroll);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, /*optional*/::mlir::ArrayAttr unroll);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/::mlir::ArrayAttr unroll);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    static StringRef getUnrollAttrName() { return "unroll"; }
    Index getUnroll() {
      Index result;
      ArrayAttr unrollAttr = unroll().getValue();
      for (auto &elem : unrollAttr) 
        result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
      return result;
    }
    unsigned getUnrollFac() {
      unsigned factor = 1;
      if (unroll().hasValue()) {
        Index unroll = getUnroll();
        factor = std::accumulate(unroll.begin(), unroll.end(), 1,
                                      std::multiplies<int64_t>());
      }
      return factor;
    }
    size_t getUnrollDim() {
      size_t dimension = 0;
      if (unroll().hasValue()) {
        auto unroll = getUnroll();
        auto unrollFac = getUnrollFac();
        auto it = llvm::find_if(unroll, [&](int64_t x) {
          return x == unrollFac;
        });
        assert(it != unroll.end() &&
              "expected to find unroll factor");
        dimension = std::distance(unroll.begin(), it);
      }
      return dimension;
    } 
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::StoreOp declarations
//===----------------------------------------------------------------------===//

class StoreOpAdaptor {
public:
  StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  StoreOpAdaptor(StoreOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::Value field();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ub();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class StoreOp : public ::mlir::Op<StoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ShapeInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value temp();
  ::mlir::Value field();
  ::mlir::MutableOperandRange tempMutable();
  ::mlir::MutableOperandRange fieldMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr lbAttr();
  ::mlir::ArrayAttr lb();
  ::mlir::ArrayAttr ubAttr();
  ::mlir::ArrayAttr ub();
  void lbAttr(::mlir::ArrayAttr attr);
  void ubAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value temp, Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value temp, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value temp, ::mlir::Value field, ::mlir::ArrayAttr lb, ::mlir::ArrayAttr ub);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  void updateArgumentTypes();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  
};
} // namespace Vtc
} // namespace mlir
namespace mlir {
namespace Vtc {

//===----------------------------------------------------------------------===//
// ::mlir::Vtc::StoreResultOp declarations
//===----------------------------------------------------------------------===//

class StoreResultOpAdaptor {
public:
  StoreResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  StoreResultOpAdaptor(StoreResultOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class StoreResultOp : public ::mlir::Op<StoreResultOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreResultOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    Optional<SmallVector<OpOperand *, 10>> getReturnOpOperands();
  
};
} // namespace Vtc
} // namespace mlir

#endif  // GET_OP_CLASSES

