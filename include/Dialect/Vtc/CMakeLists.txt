add_mlir_dialect(VtcOps Vtc)
# add_mlir_interface(ShapeInterface)


set(LLVM_TARGET_DEFINITIONS ShapeInterface.td)
mlir_tablegen(ShapeInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRShapeInterfaceIncGen)
add_dependencies(mlir-headers MLIRShapeInterfaceIncGen)

add_mlir_doc(VtcDialect -gen-dialect-doc VtcDialect Vtc/)
add_mlir_doc(ShapeInterface -gen-op-interface-doc ShapeInterface Vtc/)
add_mlir_doc(VtcOps -gen-op-doc VtcOps Vtc/)
