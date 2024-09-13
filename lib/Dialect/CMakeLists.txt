# add_mlir_interface_library(ShapeInterface)
add_mlir_dialect_library(Vtc 
VtcUtils.cpp
VtcTypes.cpp
VtcDialect.cpp
VtcOps.cpp
ShapeInterface.cpp 
OffsetInterface.cpp
ShiftInterface.cpp
ExtentInterface.cpp
ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/Dialect/Vtc  
DEPENDS
MLIRVtcOpsIncGen
# MLIRVtcPassIncGen
MLIRShapeInterfaceIncGen
MLIROffsetInterfaceIncGen
MLIRShiftInterfaceIncGen
MLIRExtentInterfaceIncGen
LINK_LIBS PUBLIC
MLIRIR)