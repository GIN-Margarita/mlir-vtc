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
    void (*shiftByOffset)(::mlir::Operation *, mlir::ArrayRef<int64_t>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{shiftByOffset} {}

    static inline void shiftByOffset(::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<int64_t> offset);
  };
};
} // end namespace detail
class ShiftInterface : public ::mlir::OpInterface<ShiftInterface, detail::ShiftInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShiftInterface, detail::ShiftInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ShiftInterfaceTrait : public ::mlir::OpInterface<ShiftInterface, detail::ShiftInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    void shiftByOffset(mlir::ArrayRef<int64_t> offset) {
      mlir::SmallVector<mlir::Attribute,mlir::Vtc::kIndexSize> offsetAttr;
                for( auto elem : llvm::zip(offset,(*static_cast<ConcreteOp *>(this)).offset()))
                {
                    offsetAttr.push_back(
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get((*static_cast<ConcreteOp *>(this)).getContext(),64),
                            std::get<0>(elem) + std::get<1>(elem).template cast<mlir::IntegerAttr>().getValue().getSExtValue()
                        )
                    );
                }
                (*static_cast<ConcreteOp *>(this)).offsetAttr(mlir::ArrayAttr::get(offsetAttr,(*static_cast<ConcreteOp *>(this)).getContext()));
    }
  };
  template <typename ConcreteOp>
  struct Trait : public ShiftInterfaceTrait<ConcreteOp> {};
  void shiftByOffset(mlir::ArrayRef<int64_t> offset);
};
template<typename ConcreteOp>
void detail::ShiftInterfaceInterfaceTraits::Model<ConcreteOp>::shiftByOffset(::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<int64_t> offset) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).shiftByOffset(offset);
}
