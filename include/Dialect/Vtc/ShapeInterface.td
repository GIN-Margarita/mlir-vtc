#ifndef Shape_INTERFACE
#define Shape_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInterface : OpInterface<"ShapeInterface"> 
{
      let description = [{
    Interface to get the operation bounds.
  }];
//   let cppNamespace = "::mlir::Vtc";

  let methods =[
    InterfaceMethod<"getBounds",
    "mlir::Vtc::Index",
    "getLB",
    (ins),
    [{
        mlir::Vtc::Index result;
        mlir::Optional<mlir::ArrayAttr> lb = $_op.lb();
        for(auto & elem : lb.getValue())
            result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
        return result;
    }]
  >,
    InterfaceMethod<"Get the upper bound of the operation",
        "mlir::Vtc::Index",
        "getUB",
        (ins),
        [{
            mlir::Vtc::Index result;
            mlir::Optional<mlir::ArrayAttr> ub = $_op.ub();
            for(auto & elem : ub.getValue())
                result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
            return result;
        }]
    >,
    InterfaceMethod<"clear the optional range atrributes",
        "void",
        "cleanInferredShape",
        (ins),
        [{
            if(isOptionalArrayAttr($_op.lb()) && isOptionalArrayAttr($_op.ub())){
                $_op.removeAttr("$_op.getLBAttrName()");
                $_op.removeAttr("$_op.getUBAttrName()");
                //Update the result types

                for(auto result: $_op.getOperation()->getResults())
                {
                    auto oldType = result.getType().template cast<mlir::Vtc::GridType>();
                    result.setType(mlir::Vtc::TempType::get(oldType.getElementType(),oldType.getAllocation()));
                }
            }
        }]
    >,
    InterfaceMethod<"Set the shape of operation",
        "void",
        "updateShape",
        (ins "mlir::ArrayRef<int64_t>":$lb,"mlir::ArrayRef<int64_t>":$ub),
        [{
            using namespace mlir;
            SmallVector<Attribute,mlir::Vtc::kIndexSize> lbAttr,ubAttr;

            llvm::transform(lb,std::back_inserter(lbAttr),[&](int64_t elem){return IntegerAttr::get(IntegerType::get($_op.getContext(),64 ),elem);});
            llvm::transform(ub,std::back_inserter(lbAttr),[&](int64_t elem){return IntegerAttr::get(IntegerType::get($_op.getContext(),64 ),elem);});

            $_op.lbAttr(ArrayAttr::get(lbAttr,$_op.getContext()));
            $_op.ubAttr(ArrayAttr::get(ubAttr,$_op.getContext()));

            //Update the result types
            for(auto result: $_op.getOperation()->getResults())
            {
                auto oldType = result.getType().template cast<mlir::Vtc::GridType>();
                result.setType(mlir::Vtc::TempType::get(oldType.getElementType(),oldType.getAllocation(),lb,ub));
             }
        }]
    >,
    InterfaceMethod<
        "Verify if the operation has vaild bounds",
        "bool",
        "hasShape",
        (ins),
        [{
            mlir::Optional<mlir::ArrayAttr> lb = $_op.lb();
            mlir::Optional<mlir::ArrayAttr> ub = $_op.ub();
            return lb.hasValue() && ub.hasValue();
        }]
    >,
    InterfaceMethod<
        "Get the rank of the operation",
        "unsigned",
        "getRank",
        (ins),
        [{
            mlir::Optional<mlir::ArrayAttr> lb = $_op.lb();
            mlir::Optional<mlir::ArrayAttr> ub = $_op.ub();
            
            assert(lb.getValue().size() == ub.getValue().size() && "expected the same size");
            return (unsigned) lb.getValue().size();
        }]
    >,
    InterfaceMethod<
        "Update the argument types",
        "void",
        "updateArgumentTypes",
        (ins),
        [{}]
    >
  ];

    // let extraClassDeclaration = [{
    //     mlir::Vtc::Index getLB();
    // }];
  
}

#endif
