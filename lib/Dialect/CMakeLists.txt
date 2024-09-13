# add_mlir_interface_library(ShapeInterface)
add_mlir_dialect_library(Vtc 
VtcDialect.cpp
VtcOps.cpp
VtcTypes.cpp
ShapeInterface.cpp 
ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/Dialect/Vtc  
DEPENDS
MLIRVtcOpsIncGen
# MLIRVtcPassIncGen
MLIRShapeInterfaceIncGen

LINK_LIBS PUBLIC
MLIRIR)