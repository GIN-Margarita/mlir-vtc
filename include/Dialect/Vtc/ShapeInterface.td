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
  >];

    // let extraClassDeclaration = [{
    //     mlir::Vtc::Index getLB();
    // }];
  
}

#endif
