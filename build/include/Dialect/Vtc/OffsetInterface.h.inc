/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class OffsetInterface;
namespace detail {
struct OffsetInterfaceInterfaceTraits {
  struct Concept {
    mlir::Vtc::Index (*getOffset)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getOffset} {}

    static inline mlir::Vtc::Index getOffset(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class OffsetInterface : public ::mlir::OpInterface<OffsetInterface, detail::OffsetInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OffsetInterface, detail::OffsetInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct OffsetInterfaceTrait : public ::mlir::OpInterface<OffsetInterface, detail::OffsetInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public OffsetInterfaceTrait<ConcreteOp> {};
  mlir::Vtc::Index getOffset();
};
template<typename ConcreteOp>
mlir::Vtc::Index detail::OffsetInterfaceInterfaceTraits::Model<ConcreteOp>::getOffset(::mlir::Operation *tablegen_opaque_val) {
  mlir::Vtc::Index result;
            mlir::ArrayAttr offset = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).offset();
            for (auto & elem : offset){
                result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
            return result;
            }
}
