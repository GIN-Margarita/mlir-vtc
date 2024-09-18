#pragma once 

#include "mlir/Pass/Pass.h"
namespace 
mlir {

std::unique_ptr<OperationPass<FuncOp>> createDomainSplitPass();   


std::unique_ptr<OperationPass<FuncOp>> createVtcInliningPass();

std::unique_ptr<OperationPass<FuncOp>> createVtcUnrollingPass();

std::unique_ptr<OperationPass<FuncOp>> createCombineToIfElsePass();

std::unique_ptr<OperationPass<FuncOp>> createShapeInferencePass();

std::unique_ptr<OperationPass<FuncOp>> createShapeOverlapPass();

std::unique_ptr<OperationPass<FuncOp>> createStorageMaterializationPass();

std::unique_ptr<OperationPass<FuncOp>> createPeelOddIterationsPass();

#define  GEN_PASS_REGISTRATION
#include "Dialect/Vtc/Passes.h.inc"

#define  GEN_PASS_CLASSES
#include "Dialect/Vtc/Passes.h.inc"



}