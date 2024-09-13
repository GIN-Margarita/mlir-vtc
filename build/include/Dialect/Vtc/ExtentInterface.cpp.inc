/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

mlir::Value ExtentInterface::getTemp() {
      return getImpl()->getTemp(getOperation());
  }
std::tuple<mlir::Vtc::Index,mlir::Vtc::Index> ExtentInterface::getAccessExtent() {
      return getImpl()->getAccessExtent(getOperation());
  }
