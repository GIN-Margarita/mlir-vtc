cmake_minimum_required(VERSION 3.10)

project(mlir-vtc LANGUAGES CXX C)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ 14 standard")

set(LLVM_LIT_ARGS "-sv" CACHE STRING "LLVM lit args")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

#link LLVMDIR
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
#add sanitizers
include(sanitizers)

set(CUDA_BACKEND_ENABLED 1 CACHE BOOL "Enable building the oec CUDA backend")
set(ROCM_BACKEND_ENABLED 0 CACHE BOOL "Enable building the oec ROCM backend")
if(CUDA_BACKEND_ENABLED)
  add_definitions(-DCUDA_BACKEND_ENABLED)
endif()
if(ROCM_BACKEND_ENABLED)
  add_definitions(-DROCM_BACKEND_ENABLED)
endif()

if(CUDA_BACKEND_ENABLED)
    if(NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
        message(FATAL_ERROR "CUDA backend requires NVPTX")
    endif()

    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(SEND_ERROR "Building the oec CUDA backend requires CUDA")
    endif()
  find_library(CUDA_RUNTIME_LIBRARY cuda)
endif()

add_subdirectory(include)
add_subdirectory(lib)