#ifndef DIALECT_vtc_PASSES
#define DIALECT_vtc_PASSES

include "mlir/Pass/PassBase.td"

def DomainSplitPass : FunctionPass<"vtc-domain-split"> {
  let summary = "Copy apply ops used in multiple domains";
  let constructor = "mlir::createDomainSplitPass()";
}

def VtcInliningPass : FunctionPass<"vtc-inlining"> {
  let summary = "Inline vtc apply ops";
  let constructor = "mlir::createVtcInliningPass()";
}

def VtcUnrollingPass : FunctionPass<"vtc-unrolling"> {
  let summary = "Unroll vtc apply ops";
  let constructor = "mlir::createVtcUnrollingPass()";
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"2",
           "Number of unrolled loop iterations">,
    Option<"unrollIndex", "unroll-index", "unsigned", /*default=*/"1",
           "Unroll index specifying the unrolling dimension">,
  ];
}

def CombineToIfElsePass : FunctionPass<"vtc-combine-to-ifelse"> {
  let summary = "Lower vtc combine ops to if/else";
  let constructor = "mlir::createCombineToIfElsePass()";
  let options = [
    Option<"prepareOnly", "prepare-only", "bool", /*default=*/"false", 
           "Lower extra operands and fuse multiple producers attached to one combine">,
    Option<"internalOnly", "internal-only", "bool", /*default=*/"false", 
           "Lower only combine ops embedded in between apply ops">,
  ];
}

def ShapeInferencePass : FunctionPass<"vtc-shape-inference"> {
  let summary = "Infer loop bounds and storage shapes";
  let constructor = "mlir::createShapeInferencePass()";
  let options = [
    Option<"extendStorage", "extend-storage", "bool", /*default=*/"false", 
           "Extend the shape of store and buffer ops">,
  ];
}

def ShapeOverlapPass : FunctionPass<"vtc-shape-overlap"> {
  let summary = "Handle overlapping stores";
  let constructor = "mlir::createShapeOverlapPass()";
}

def StorageMaterializationPass : FunctionPass<"vtc-storage-materialization"> {
  let summary = "Introduce explicit storage between combine and apply ops";
  let constructor = "mlir::createStorageMaterializationPass()";
}

def PeelOddIterationsPass : FunctionPass<"vtc-peel-odd-iterations"> {
  let summary = "Peel odd iterations that are not a multiple of the unroll factor";
  let constructor = "mlir::createPeelOddIterationsPass()";
}

#endif // DIALECT_vtc_PASSES
