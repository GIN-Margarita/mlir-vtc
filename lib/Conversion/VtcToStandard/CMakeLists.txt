add_mlir_dialect_library(VtcToStandard
    ConvertVtcToStandard.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Conversion/VtcToStandard

    DEPENDS
    MLIRVtcToStandardPassIncGen
)

target_link_libraries(VtcToStandard PUBLIC MLIRIR)