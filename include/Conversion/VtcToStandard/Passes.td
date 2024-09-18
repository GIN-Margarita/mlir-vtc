include "mlir/Pass/PassBase.td"

def StencilToStandardPass : Pass<"convert-vtc-to-std", "ModuleOp"> {
  let summary = "Convert vtc dialect to standard operations";
  let constructor = "mlir::createConvertVtcToStandardPass()";
}
