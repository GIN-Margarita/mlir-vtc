/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

mlir::Vtc::Index ShapeInterface::getLB() {
      return getImpl()->getLB(getOperation());
  }
mlir::Vtc::Index ShapeInterface::getUB() {
      return getImpl()->getUB(getOperation());
  }
void ShapeInterface::cleanInferredShape() {
      return getImpl()->cleanInferredShape(getOperation());
  }
void ShapeInterface::updateShape(mlir::ArrayRef<int64_t> lb, mlir::ArrayRef<int64_t> ub) {
      return getImpl()->updateShape(getOperation(), lb, ub);
  }
bool ShapeInterface::hasShape() {
      return getImpl()->hasShape(getOperation());
  }
unsigned ShapeInterface::getRank() {
      return getImpl()->getRank(getOperation());
  }
void ShapeInterface::updateArgumentTypes() {
      return getImpl()->updateArgumentTypes(getOperation());
  }
