#ifndef VTC_OPS
#define VTC_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect/Vtc/VtcBase.td"
include "Dialect/Vtc/ShapeInterface.td"


def Vtc_CastOp : Vtc_Op<"cast", [
  DeclareOpInterfaceMethods<ShapeInterface>]> {
  let summary = "cast the dynamiclly shaped input fields";
  let description = [{
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
      %0 = Vtc.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!Vtc.field<?x?x?xf64>) -> !Vtc.field<70x70x60xf64>
  }];

  let arguments = (ins Vtc_Field:$field, Vtc_Index:$lb, Vtc_Index:$ub);
  let results = (outs Vtc_Field:$res);

  let hasCanonicalizer = 1;

  let builders = [
    OpBuilderDAG<(ins "Value":$field, "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), 
    [{
      $_state.addOperands(field);
      $_state.addAttribute(getLBAttrName(), $_builder.getI64ArrayAttr(lb));
      $_state.addAttribute(getUBAttrName(), $_builder.getI64ArrayAttr(ub));
      $_state.addTypes(Vtc::FieldType::get(
        field.getType().cast<Vtc::GridType>().getElementType(), 
        applyFunElementWise(lb, ub, std::minus<int64_t>())));
    }]>
  ];

  let assemblyFormat = [{
    $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let verifier = [{
    auto fieldType = field().getType().cast<Vtc::GridType>();
    auto resType = res().getType().cast<Vtc::GridType>();
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if(!fieldType.hasDynamicShape())
      return emitOpError("expected field to have dynamic shape");
    if(resType.hasDynamicShape())
      return emitOpError("expected result to have static shape");
    if(fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("expected the field and result types to have the same allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and result types have different element types");
    if(shapeOp.getRank() != fieldType.getRank() || shapeOp.getRank() != resType.getRank())
      return emitOpError("expected op and the field and result types to have the same rank");
    
    // Ensure the shape matches the result type
    if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected op and result type to have the same shape");

    // Verify all users fit the shape
    for(auto user : res().getUsers()) {
      if(auto userOp = dyn_cast<ShapeInterface>(user)) {
        if(userOp.hasShape() &&
           (shapeOp.getLB() != applyFunElementWise(shapeOp.getLB(), userOp.getLB(), min) ||
            shapeOp.getUB() != applyFunElementWise(shapeOp.getUB(), userOp.getUB(), max))) 
          return emitOpError("shape not large enough to fit all accesses");
      }
    }

    return success();
  }];
  
  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}
#endif
