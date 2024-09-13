add_mlir_dialect(VtcOps Vtc)
# add_mlir_interface(ShapeInterface)


set(LLVM_TARGET_DEFINITIONS ShapeInterface.td)
mlir_tablegen(ShapeInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRShapeInterfaceIncGen)
add_dependencies(mlir-headers MLIRShapeInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS OffsetInterface.td)
mlir_tablegen(OffsetInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(OffsetInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIROffsetInterfaceIncGen)
add_dependencies(mlir-headers MLIROffsetInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS ShiftInterface.td)  
mlir_tablegen(ShiftInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShiftInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRShiftInterfaceIncGen)
add_dependencies(mlir-headers MLIRShiftInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS ExtentInterface.td)
mlir_tablegen(ExtentInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ExtentInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRExtentInterfaceIncGen)
add_dependencies(mlir-headers MLIRExtentInterfaceIncGen)


add_mlir_doc(VtcDialect -gen-dialect-doc VtcDialect Vtc/)
add_mlir_doc(ShapeInterface -gen-op-interface-doc ShapeInterface Vtc/)
add_mlir_doc(VtcOps -gen-op-doc VtcOps Vtc/)
