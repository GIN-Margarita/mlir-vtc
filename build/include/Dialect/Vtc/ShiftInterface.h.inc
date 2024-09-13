/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class ShiftInterface;
namespace detail {
struct ShiftInterfaceInterfaceTraits {
  struct Concept {
    void (*ShiftByOffset)(::mlir::Operation *, mlir::ArrayRef<int64_t>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{ShiftByOffset} {}

    static inline void ShiftByOffset(::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<int64_t> offset);
  };
};
} // end namespace detail
class ShiftInterface : public ::mlir::OpInterface<ShiftInterface, detail::ShiftInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShiftInterface, detail::ShiftInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ShiftInterfaceTrait : public ::mlir::OpInterface<ShiftInterface, detail::ShiftInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ShiftInterfaceTrait<ConcreteOp> {};
  void ShiftByOffset(mlir::ArrayRef<int64_t> offset);
};
template<typename ConcreteOp>
void detail::ShiftInterfaceInterfaceTraits::Model<ConcreteOp>::ShiftByOffset(::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<int64_t> offset) {
  mlir::SmallVector<mlir::Attribute,mlir::Vtc::kIndexSize> offsetAttr;
                for( auto elem : llvm::zip(offset,(llvm::cast<ConcreteOp>(tablegen_opaque_val)).offset()))
                {
                    offsetAttr.push_back(
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext(),64),
                            std::get<0>(elem) + std::get<1>(elem).template cast<mlir::IntegerAttr>().getValue().getSExtValue()
                        )
                    );
                }
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).offsetAttr(mlir::ArrayAttr::get(offsetAttr,(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext()));
}
