include "mlir/IR/OpBase.td"

def ExtentInterface : OpInterface<"ExtentInterface">
{
    let description = [{
        Interface to get the accessed temporary
    }];
    let methods = [
        InterfaceMethod<
            "Get the accessed temporary",
            "mlir::Value",
            "getTemp",
            (ins),
            [{
                return $_op.temp();
            }]
        >,
        InterfaceMethod<
            "Get the access extent",
            "std::tuple<mlir::Vtc::Index,mlir::Vtc::Index>",
            "getAccessExtent",
            (ins),
            [{}],
            [{
                mlir::Vtc::Index result;
                mlir::ArrayAttr offset = $_op.offset();
                for (auto &elem : offset.getValue())
                result.push_back(elem.cast<mlir::IntegerAttr>().getValue().getSExtValue());
                return std::make_tuple(result, result);
            }]
        >
    ];
}