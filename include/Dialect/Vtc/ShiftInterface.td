include "mlir/IR/OpBase.td"

def ShiftInterface : OpInterface<"ShiftInterface">
{
    let description = [{
            interface to shift the operation offset
    }];

    let methods = [
        InterfaceMethod<
            "Shift the operation by a constant offset",
            "void",
            "ShiftByOffset",
            (ins "mlir::ArrayRef<int64_t>":$offset),
            [{
                mlir::SmallVector<mlir::Attribute,mlir::Vtc::kIndexSize> offsetAttr;
                for( auto elem : llvm::zip(offset,$_op.offset()))
                {
                    offsetAttr.push_back(
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get($_op.getContext(),64),
                            std::get<0>(elem) + std::get<1>(elem).template cast<mlir::IntegerAttr>().getValue().getSExtValue()
                        )
                    );
                }
                $_op.offsetAttr(mlir::ArrayAttr::get(offsetAttr,$_op.getContext()));
            }]
        >
    ];
}