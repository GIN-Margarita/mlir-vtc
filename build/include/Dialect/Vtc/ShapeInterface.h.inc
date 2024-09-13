/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class ShapeInterface;
namespace detail {
struct ShapeInterfaceInterfaceTraits {
  struct Concept {
    mlir::Vtc::Index (*getLB)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getLB} {}

    static inline mlir::Vtc::Index getLB(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class ShapeInterface : public ::mlir::OpInterface<ShapeInterface, detail::ShapeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShapeInterface, detail::ShapeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ShapeInterfaceTrait : public ::mlir::OpInterface<ShapeInterface, detail::ShapeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ShapeInterfaceTrait<ConcreteOp> {};
  mlir::Vtc::Index getLB();
};
template<typename ConcreteOp>
mlir::Vtc::Index detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::getLB(::mlir::Operation *tablegen_opaque_val) {
  mlir::Vtc::Index result;
        mlir::Optional<mlir::ArrayAttr> lb = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).lb();
        for(auto & elem : lb.getValue())
            result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
        return result;
}
