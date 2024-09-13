/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class ExtentInterface;
namespace detail {
struct ExtentInterfaceInterfaceTraits {
  struct Concept {
    mlir::Value (*getTemp)(::mlir::Operation *);
    std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> (*getAccessExtent)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getTemp, getAccessExtent} {}

    static inline mlir::Value getTemp(::mlir::Operation *tablegen_opaque_val);
    static inline std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> getAccessExtent(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class ExtentInterface : public ::mlir::OpInterface<ExtentInterface, detail::ExtentInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ExtentInterface, detail::ExtentInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ExtentInterfaceTrait : public ::mlir::OpInterface<ExtentInterface, detail::ExtentInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> getAccessExtent() {
      using namespace mlir;
                mlir::Vtc::Index result;
                ArrayAttr offset = (*static_cast<ConcreteOp *>(this)).offset();
                for (auto &elem : offset.getValue())
                result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
                return std::make_tuple(result, result);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public ExtentInterfaceTrait<ConcreteOp> {};
  mlir::Value getTemp();
  std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> getAccessExtent();
};
template<typename ConcreteOp>
mlir::Value detail::ExtentInterfaceInterfaceTraits::Model<ConcreteOp>::getTemp(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).temp();
}
template<typename ConcreteOp>
std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> detail::ExtentInterfaceInterfaceTraits::Model<ConcreteOp>::getAccessExtent(::mlir::Operation *tablegen_opaque_val) {
  
}
