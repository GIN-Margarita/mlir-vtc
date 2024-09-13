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
    mlir::Vtc::Index (*getUB)(::mlir::Operation *);
    void (*cleanInferredShape)(::mlir::Operation *);
    void (*updateShape)(::mlir::Operation *, mlir::ArrayRef<int64_t>, mlir::ArrayRef<int64_t>);
    bool (*hasShape)(::mlir::Operation *);
    unsigned (*getRank)(::mlir::Operation *);
    void (*updateArgumentTypes)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getLB, getUB, cleanInferredShape, updateShape, hasShape, getRank, updateArgumentTypes} {}

    static inline mlir::Vtc::Index getLB(::mlir::Operation *tablegen_opaque_val);
    static inline mlir::Vtc::Index getUB(::mlir::Operation *tablegen_opaque_val);
    static inline void cleanInferredShape(::mlir::Operation *tablegen_opaque_val);
    static inline void updateShape(::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<int64_t> lb, mlir::ArrayRef<int64_t> ub);
    static inline bool hasShape(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getRank(::mlir::Operation *tablegen_opaque_val);
    static inline void updateArgumentTypes(::mlir::Operation *tablegen_opaque_val);
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
  mlir::Vtc::Index getUB();
  void cleanInferredShape();
  void updateShape(mlir::ArrayRef<int64_t> lb, mlir::ArrayRef<int64_t> ub);
  bool hasShape();
  unsigned getRank();
  void updateArgumentTypes();
};
template<typename ConcreteOp>
mlir::Vtc::Index detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::getLB(::mlir::Operation *tablegen_opaque_val) {
  mlir::Vtc::Index result;
        mlir::Optional<mlir::ArrayAttr> lb = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).lb();
        for(auto & elem : lb.getValue())
            result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
        return result;
}
template<typename ConcreteOp>
mlir::Vtc::Index detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::getUB(::mlir::Operation *tablegen_opaque_val) {
  mlir::Vtc::Index result;
            mlir::Optional<mlir::ArrayAttr> ub = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).ub();
            for(auto & elem : ub.getValue())
                result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
            return result;
}
template<typename ConcreteOp>
void detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::cleanInferredShape(::mlir::Operation *tablegen_opaque_val) {
  if(isOptionalArrayAttr((llvm::cast<ConcreteOp>(tablegen_opaque_val)).lb()) && isOptionalArrayAttr((llvm::cast<ConcreteOp>(tablegen_opaque_val)).ub())){
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeAttr("(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLBAttrName()");
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeAttr("(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getUBAttrName()");
                //Update the result types

                for(auto result: (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getResults())
                {
                    auto oldType = result.getType().template cast<mlir::Vtc::GridType>();
                    result.setType(mlir::Vtc::TempType::get(oldType.getElementType(),oldType.getAllocation()));
                }
            }
}
template<typename ConcreteOp>
void detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::updateShape(::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<int64_t> lb, mlir::ArrayRef<int64_t> ub) {
  using namespace mlir;
            SmallVector<Attribute,mlir::Vtc::kIndexSize> lbAttr,ubAttr;

            llvm::transform(lb,std::back_inserter(lbAttr),[&](int64_t elem){return IntegerAttr::get(IntegerType::get((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext(),64 ),elem);});
            llvm::transform(ub,std::back_inserter(lbAttr),[&](int64_t elem){return IntegerAttr::get(IntegerType::get((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext(),64 ),elem);});

            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).lbAttr(ArrayAttr::get(lbAttr,(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext()));
            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).ubAttr(ArrayAttr::get(ubAttr,(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext()));

            //Update the result types
            for(auto result: (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getResults())
            {
                auto oldType = result.getType().template cast<mlir::Vtc::GridType>();
                result.setType(mlir::Vtc::TempType::get(oldType.getElementType(),oldType.getAllocation(),lb,ub));
             }
}
template<typename ConcreteOp>
bool detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::hasShape(::mlir::Operation *tablegen_opaque_val) {
  mlir::Optional<mlir::ArrayAttr> lb = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).lb();
            mlir::Optional<mlir::ArrayAttr> ub = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).ub();
            return lb.hasValue() && ub.hasValue();
}
template<typename ConcreteOp>
unsigned detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::getRank(::mlir::Operation *tablegen_opaque_val) {
  mlir::Optional<mlir::ArrayAttr> lb = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).lb();
            mlir::Optional<mlir::ArrayAttr> ub = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).ub();
            
            assert(lb.getValue().size() == ub.getValue().size() && "expected the same size");
            return (unsigned) lb.getValue().size();
}
template<typename ConcreteOp>
void detail::ShapeInterfaceInterfaceTraits::Model<ConcreteOp>::updateArgumentTypes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).updateArgumentTypes();
}
