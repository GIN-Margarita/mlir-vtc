include "mlir/IR/OpBase.td"

def OffsetInterface : OpInterface<"OffsetInterface">{
    let description = [{
        Interface to get the operation offset.
    }];

    let methods = [
         InterfaceMethod<"Get the offset of operation",
        "mlir::Vtc::Index",
        "getOffset",
        (ins),
        [{
            mlir::Vtc::Index result;
            mlir::ArrayAttr offset = $_op.offset();
            for (auto & elem : offset){
                result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
            return result;
            }
        }]
        >
    ];
    //
}