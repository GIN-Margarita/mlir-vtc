#ifndef VTC_OPS
#define VTC_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect/Vtc/VtcBase.td"
include "Dialect/Vtc/ShapeInterface.td"
include "Dialect/Vtc/OffsetInterface.td"
include "Dialect/Vtc/ShiftInterface.td"
include "Dialect/Vtc/ExtentInterface.td"

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

def Vtc_IndexOp : Vtc_Op<"index",
[
  DeclareOpInterfaceMethods<OffsetInterface>,
  DeclareOpInterfaceMethods<ShiftInterface>,
  NoSideEffect
]>{
    let summary  = "index of operation";

    let description = [{
        return index of loop iteration
    }];

    let arguments = (ins Confined<I64Attr,[IntMinValue<0>,IntMaxValue<2>]>:$dim,
                     Vtc_Index:$offset);

    let results = (outs Index:$idx);

    
    let builders = [
        OpBuilderDAG<(ins "int64_t":$dim,"ArrayRef<int64_t>":$offset),[{
            $_state.addAttribute(getDimAttrName(), $_builder.getI64IntegerAttr(dim));
            $_state.addAttribute(getOffsetAttrName(), $_builder.getI64ArrayAttr(offset));
            $_state.addTypes($_builder.getIndexType());
        }]>
    ];

    let assemblyFormat = [{
            $dim $offset attr-dict-with-keyword `:` type($idx)
    }];

    let verifier = [{
        return success();
    }];

    let extraClassDeclaration = [{
        static StringRef getDimAttrName() { return "dim"; }
        static StringRef getOffsetAttrName() { return "offset"; }
    }];
}

def Vtc_AccessOp : Vtc_Op<"access",[
    DeclareOpInterfaceMethods<OffsetInterface>,
    DeclareOpInterfaceMethods<ShiftInterface>,
    DeclareOpInterfaceMethods<ExtentInterface>,
    NoSideEffect
]>
{
    let summary = "access operation";

    let description = [{
        return access of loop iteration
    }];
    
    let arguments = (ins Vtc_Temp:$temp,Vtc_Index:$offset);
    let results = (outs Vtc_Element:$res);

    let builders = [
        OpBuilderDAG<(ins "Value":$temp,"ArrayRef<int64_t>":$offset),
        [{
            $_state.addOperands(temp);
            $_state.addAttribute(getOffsetAttrName(), $_builder.getI64ArrayAttr(offset));
            $_state.addTypes(temp.getType().cast<Vtc::GridType>().getElementType());
        }]>
    ];
    
    let assemblyFormat = [{
        $temp `(` $offset `)` attr-dict-with-keyword `:` functional-type($temp, $res)
    }];

    let verifier = [{
        auto tempType = temp().getType().cast<Vtc::GridType>();

        if(offset().size() != tempType.getRank())
            return emitOpError("expected offset to have the same rank as the temp type");
        if(res().getType() != tempType.getElementType())
            return emitOpError("expected result to have the same element type as the temp type");

        return success();
    }];

    let extraClassDeclaration =[{
        static StringRef getOffsetAttrName() { return "offset"; }
    }];
}

def Vtc_DynAccessOp : Vtc_Op<"dyn_access",[
    DeclareOpInterfaceMethods<ShiftInterface,["shiftByOffset"]>,
    DeclareOpInterfaceMethods<ExtentInterface,["getAccessExtent"]>,
    NoSideEffect
]>{
      let summary = "Vtc dyn_access operation";
  let description = [{
    This operation accesses a temporary element given a dynamic offset. 
    The offset is specified in absolute coordinates. An additional 
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation. 

    Example:
      %0 = Vtc.dyn_access %temp (%i, %j, %k) in [-1, -1, -1] : [1, 1, 1] : !Vtc.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Vtc_Temp:$temp, 
                       Variadic<Index>:$offset, 
                       Vtc_Index:$lb, 
                       Vtc_Index:$ub);
  let results = (outs Vtc_Element:$res);
  
  let builders = [
    OpBuilderDAG<(ins "Value":$temp, "ValueRange":$offset, "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), 
    [{
      $_state.addOperands(temp);
      $_state.addOperands(offset);
      $_state.addAttribute(getLBAttrName(), $_builder.getI64ArrayAttr(lb));
      $_state.addAttribute(getUBAttrName(), $_builder.getI64ArrayAttr(ub));
      auto tempType = temp.getType().cast<Vtc::GridType>();
      $_state.addTypes(tempType.getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $temp `(` $offset `)` `in` $lb `:` $ub attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let verifier = [{
    auto tempType = temp().getType().cast<Vtc::GridType>();
    if (offset().size() != tempType.getRank())
      return emitOpError("offset and temp dimensions do not match");
    if (res().getType() != tempType.getElementType())
      return emitOpError("result type and element type are inconsistent");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Vtc_LoadOp : Vtc_Op<"load",[
    DeclareOpInterfaceMethods<ShapeInterface>,
    NoSideEffect
]>
{
    let summary = "load operation";
  let description = [{
    This operation takes a field and returns a temporary values.

    Example:
      %0 = Vtc.load %field : (!Vtc.field<70x70x60xf64>) -> !Vtc.temp<?x?x?xf64>
  }];

  let arguments = (ins Vtc_Field:$field,
                       OptionalAttr<Vtc_Index>:$lb, 
                       OptionalAttr<Vtc_Index>:$ub);
  let results = (outs Vtc_Temp:$res);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilderDAG<(ins "Value":$field, "Optional<ArrayAttr>":$lb, "Optional<ArrayAttr>":$ub), 
    [{
      $_state.addOperands(field);
      auto fieldType = field.getType().cast<Vtc::GridType>();
      if(lb.hasValue() && ub.hasValue()) {
        $_state.addAttribute(getLBAttrName(), lb.getValue());
        $_state.addAttribute(getUBAttrName(), ub.getValue()); 
        Index lower, upper;
        for (auto &elem : lb.getValue()) 
          lower.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        for (auto &elem : ub.getValue()) 
          upper.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        $_state.addTypes(Vtc::TempType::get(fieldType.getElementType(), fieldType.getAllocation(), lower, upper));
      } else {
        $_state.addTypes(Vtc::TempType::get(fieldType.getElementType(), fieldType.getShape()));
      }
    }]>
  ];

  let assemblyFormat = [{
    $field (`(` $lb^ `:` $ub `)`)? attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let verifier = [{
    // Check the field and result types
    auto fieldType = field().getType().cast<Vtc::GridType>();
    auto resType = res().getType().cast<Vtc::GridType>();
    if(fieldType.hasDynamicShape())
      return emitOpError("expected fields to have static shape");
    if (fieldType.getRank() != resType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and temp types have different element types");
    
    // Ensure the shape matches the field and result types
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if(shapeOp.hasShape()) {
      if (!fieldType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected the field type to be larger than the op shape");
      if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected op and result type to have the same shape");
    }

    if(!isa<Vtc::CastOp>(field().getDefiningOp()))
      return emitOpError("expected the defining op of the field is a cast operation");

    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}


def Vtc_BufferOp : Vtc_Op<"buffer",
[
    DeclareOpInterfaceMethods<ShapeInterface>,
    NoSideEffect
]>{
     let summary = "buffer operation";
  let description = [{
    Prevents fusion of consecutive Vtc.apply operations.

    Example:
      %0 = Vtc.buffer %buffered : (!Vtc.temp<?x?x?xf64>) -> !Vtc.temp<?x?x?xf64>
  }];

  let arguments = (ins Vtc_Temp:$temp,
                       OptionalAttr<Vtc_Index>:$lb,
                       OptionalAttr<Vtc_Index>:$ub);
  let results = (outs Vtc_Temp:$res);

  let builders = [
    OpBuilderDAG<(ins "Value":$temp), 
    [{
      $_state.addOperands(temp);
      $_state.addTypes(temp.getType());
    }]>
  ];

  let assemblyFormat = [{
    $temp (`(` $lb^ `:` $ub`)`)? attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let verifier = [{
    // Check the temp and result types
    auto tempType = temp().getType().cast<Vtc::GridType>();
    auto resType = res().getType().cast<Vtc::GridType>();
    if (resType.getRank() != tempType.getRank())
      return emitOpError("the result and temp types have different rank");
    if (resType.getAllocation() != tempType.getAllocation())
      return emitOpError("the result and temp types have different allocation");
    if (resType.getElementType() != tempType.getElementType())
      return emitOpError("the result and temp types have different element types");

    // Ensure the shape matches the temp and result types
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if(shapeOp.hasShape()) {
      if (!tempType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected the temp type to be larger than the op shape");
      if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected op and result type to have the same shape");
    }

    if(!(isa<Vtc::ApplyOp>(temp().getDefiningOp()) ||
         isa<Vtc::CombineOp>(temp().getDefiningOp())))
      return emitOpError("expected buffer to connect to an apply or combine op");
    
    if (!llvm::all_of(temp().getUsers(), [](Operation *op) { return isa<Vtc::BufferOp>(op); })) 
      return emitOpError("expected only buffers use the same value");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}


def Vtc_StoreOp : Vtc_Op<"store", [
  DeclareOpInterfaceMethods<ShapeInterface>]> {
  let summary = "store operation";
  let description = [{
    This operation takes a temp and writes a field on a user defined range.

    Example:
      Vtc.store %temp to %field ([0,0,0] : [64,64,60]) : !Vtc.temp<?x?x?xf64> to !Vtc.field<70x70x60xf64>
  }];

  let arguments = (ins Vtc_Temp:$temp, 
                       Vtc_Field:$field, 
                       Vtc_Index:$lb, 
                       Vtc_Index:$ub);
  let results = (outs);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilderDAG<(ins "Value":$temp, "Value":$field, "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), 
    [{
      $_state.addOperands({temp, field});
      $_state.addAttribute(getLBAttrName(), $_builder.getI64ArrayAttr(lb));
      $_state.addAttribute(getUBAttrName(), $_builder.getI64ArrayAttr(ub)); 
    }]>
  ];
  
  let assemblyFormat = [{
    $temp `to` $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($temp) `to` type($field)
  }];

  let verifier = [{
    // Check the field and result types
    auto fieldType = field().getType().cast<Vtc::GridType>();
    auto tempType = temp().getType().cast<Vtc::GridType>();
    if(fieldType.hasDynamicShape())
      return emitOpError("expected fields to have static shape");
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != tempType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != tempType.getElementType())
      return emitOpError("the field and temp types have different element types");

    // Ensure the shape matches the temp and result types
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    if (!fieldType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected the field type to be larger than the op shape");
    if (!tempType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected the temp type to be larger than the op shape");

    if (!(dyn_cast<Vtc::ApplyOp>(temp().getDefiningOp()) || 
          dyn_cast<Vtc::CombineOp>(temp().getDefiningOp())))
      return emitOpError("output temp not result of an apply or a combine op");
    if (llvm::count_if(field().getUsers(), [](Operation *op) { return isa_and_nonnull<Vtc::LoadOp>(op); }) != 0)
      return emitOpError("an output cannot be an input");  
    if (llvm::count_if(field().getUsers(), [](Operation *op) { return isa_and_nonnull<Vtc::StoreOp>(op); }) != 1)
      return emitOpError("multiple stores to the same output");  
    
    if(!isa<Vtc::CastOp>(field().getDefiningOp()))
      return emitOpError("expected the defining op of the field is a cast operation");

    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Vtc_ApplyOp : Vtc_Op<"apply", [
  DeclareOpInterfaceMethods<ShapeInterface, ["updateArgumentTypes"]>,
  IsolatedFromAbove, 
  SingleBlockImplicitTerminator<"ReturnOp">,
  NoSideEffect]> {
  let summary = "apply operation";
  let description = [{
    This operation takes a Vtc function plus parameters and applies 
    the Vtc function to the output temp.

    Example:

      %0 = Vtc.apply (%arg0=%0 : !Vtc.temp<?x?x?xf64>) -> !Vtc.temp<?x?x?xf64> {
        ...
      } 
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                        OptionalAttr<Vtc_Index>:$lb, 
                        OptionalAttr<Vtc_Index>:$ub);
  let results = (outs Variadic<Vtc_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
  
  let builders = [
    OpBuilderDAG<(ins "TypeRange":$resultTypes, "ValueRange":$operands, "Optional<ArrayAttr>":$lb, "Optional<ArrayAttr>":$ub), 
    [{
      $_state.addOperands(operands);
      if(lb.hasValue() && ub.hasValue()) {
        $_state.addAttribute(getLBAttrName(), lb.getValue());
        $_state.addAttribute(getUBAttrName(), ub.getValue()); 
      }
      auto region = $_state.addRegion(); 
      region->push_back(new Block());
      region->front().addArguments(operands.getTypes());
      $_state.addTypes(resultTypes);
    }]>,
    OpBuilderDAG<(ins "TypeRange":$resultTypes, "Optional<ArrayAttr>":$lb, "Optional<ArrayAttr>":$ub), 
    [{
      if(lb.hasValue() && ub.hasValue()) {
        $_state.addAttribute(getLBAttrName(), lb.getValue());
        $_state.addAttribute(getUBAttrName(), ub.getValue()); 
      }
      $_state.addRegion();
      $_state.addTypes(resultTypes);
    }]>
  ];

  let verifier = [{
    // Check the operands
    if (region().front().getNumArguments() != operands().size())
      return emitOpError("operand and argument counts do not match");
    for (unsigned i = 0, e = operands().size(); i != e; ++i) {
      if (region().front().getArgument(i).getType() != operands()[i].getType())
        return emitOpError("operand and argument types do not match");
    }

    // Check the results
    auto shapeOp = cast<ShapeInterface>(this->getOperation());
    for(auto result : res()) {
      auto tempType = result.getType().cast<GridType>();
      if(shapeOp.hasShape()) {
        if(shapeOp.getRank() != tempType.getRank())
          return emitOpError("expected result rank to match the operation rank");
        if (!tempType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
          return emitOpError("expected temp type to have the shape of the op");
      }
    }
    return success();
  }];

  let extraClassDeclaration = [{
    bool hasOnlyEmptyStores();
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Block *getBody() { return &region().front(); }
    ShapeInterface getCombineTreeRootShape();
  }];
}

def Vtc_StoreResultOp : Vtc_Op<"store_result", 
  [NoSideEffect]> {
  let summary = "store_result operation";
  let description = [{
    The store_result operation either stores an operand value or nothing.

    Examples:
      Vtc.store_result %0 : !Vtc.result<f64>
      Vtc.store_result : !Vtc.result<f64>      
  }];

  let arguments = (ins Variadic<Vtc_Element>:$operands);
  let results = (outs Vtc_Result:$res);

  let builders = [
    OpBuilderDAG<(ins "Value":$value), 
    [{
      $_state.addOperands({value});
      $_state.addTypes(Vtc::ResultType::get(value.getType()));
    }]>
  ];

  let assemblyFormat = [{
    ($operands^)? attr-dict-with-keyword `:` functional-type($operands, $res)
  }];

  let verifier = [{
    // Check at most one operand
    if(operands().size() > 1)
      return emitOpError("expected at most one operand");

    // Check the return type
    auto resultType = res().getType().cast<Vtc::ResultType>().getResultType();
    if(operands().size() == 1 && resultType != operands()[0].getType())
      return emitOpError("operand type and result type are inconsistent");
      
    // Check the result mapping
    if(!getReturnOpOperands())
      return emitOpError("expected valid mapping to return op operands");
    return success();
  }];

  let extraClassDeclaration = [{
    Optional<SmallVector<OpOperand *, 10>> getReturnOpOperands();
  }];
}

def Vtc_ReturnOp : Vtc_Op<"return", [
  Terminator,
  HasParent<"ApplyOp">]> {
  let summary = "return operation";
  let description = [{
    The return operation terminates the the Vtc apply and writes
    the results of the Vtc operator to the temporary values returned
    by the Vtc apply operation. The types and the number of operands
    must match the results of the Vtc apply operation.

    The optional unroll attribute enables the implementation of loop
    unrolling at the Vtc dialect level.

    Examples:
      Vtc.return %0 : !Vtc.result<f64>
  }];

  let arguments = (ins Variadic<Vtc_Result>:$operands,
                   OptionalAttr<Vtc_Index>:$unroll);

  let builders = [
    OpBuilderDAG<(ins "ValueRange":$operands, "Optional<ArrayAttr>":$unroll), 
    [{
      $_state.addOperands({operands});
      if(unroll.hasValue())
        $_state.addAttribute(getUnrollAttrName(), unroll.getValue());
    }]>
  ];

  let assemblyFormat = [{
    (`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)
  }];

  let verifier = [{
    auto applyOp = cast<Vtc::ApplyOp>(getOperation()->getParentOp());
    unsigned unrollFac = getUnrollFac();

    // Verify the number of operands matches the number of apply results
    auto results = applyOp.res();
    if (getNumOperands() != unrollFac * results.size())
      return emitOpError("the operand and apply result counts do not match");
  
    // Verify the element types match
    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      auto tempType = applyOp.getResult(i).getType().cast<GridType>();
      for (unsigned j = 0; j < unrollFac; j++)
        if (getOperand(i * unrollFac + j).getType().cast<Vtc::ResultType>().getResultType() != tempType.getElementType())
          return emitOpError("the operand and apply result element types do not match");
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getUnrollAttrName() { return "unroll"; }
    Index getUnroll() {
      Index result;
      ArrayAttr unrollAttr = unroll().getValue();
      for (auto &elem : unrollAttr) 
        result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
      return result;
    }
    unsigned getUnrollFac() {
      unsigned factor = 1;
      if (unroll().hasValue()) {
        Index unroll = getUnroll();
        factor = std::accumulate(unroll.begin(), unroll.end(), 1,
                                      std::multiplies<int64_t>());
      }
      return factor;
    }
    size_t getUnrollDim() {
      size_t dimension = 0;
      if (unroll().hasValue()) {
        auto unroll = getUnroll();
        auto unrollFac = getUnrollFac();
        auto it = llvm::find_if(unroll, [&](int64_t x) {
          return x == unrollFac;
        });
        assert(it != unroll.end() &&
              "expected to find unroll factor");
        dimension = std::distance(unroll.begin(), it);
      }
      return dimension;
    } 
  }];
}

def Vtc_CombineOp : Vtc_Op<"combine", [
  DeclareOpInterfaceMethods<ShapeInterface>, 
  AttrSizedOperandSegments]> {
  let summary = "combine operation";
  let description = [{
    Combines the results computed on a lower with the results computed on 
    an upper domain. The operation combines the domain at a given index/offset 
    in a given dimension. Optional extra operands allow to combine values
    that are only written / defined on the lower or upper subdomain. The result
    values have the order upper/lower, lowerext, upperext.

    Example:
      %result = Vtc.combine 2 at 11 lower = (%0 : !Vtc.temp<?x?x?xf64>) upper = (%1 : !Vtc.temp<?x?x?xf64>) lowerext = (%2 : !Vtc.temp<?x?x?xf64>): !Vtc.temp<?x?x?xf64>, !Vtc.temp<?x?x?xf64>
  }];

  let arguments = (ins Confined<I64Attr, [IntMinValue<0>, IntMaxValue<2>]>:$dim,
                       I64Attr:$index,
                       Variadic<Vtc_Temp>:$lower,
                       Variadic<Vtc_Temp>:$upper,
                       Variadic<Vtc_Temp>:$lowerext,
                       Variadic<Vtc_Temp>:$upperext,
                       OptionalAttr<Vtc_Index>:$lb,
                       OptionalAttr<Vtc_Index>:$ub);
  let results = (outs Variadic<Vtc_Temp>:$res);
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $dim `at` $index `lower` `=` `(`$lower `:` type($lower)`)` `upper` `=` `(`$upper `:` type($upper)`)` (`lowerext` `=` `(`$lowerext^ `:` type($lowerext)`)`)? (`upperext` `=` `(`$upperext^ `:` type($upperext)`)`)? (`(` $lb^ `:` $ub`)`)? attr-dict-with-keyword `:` type($res)
  }];

  let verifier = [{
    return ::verify(*this);
  }];

  let extraClassDeclaration = [{
    int64_t getIndex() { return indexAttr().getValue().getSExtValue(); }
    SmallVector<Operation*, 4> getLowerDefiningOps() {
      SmallVector<Operation*, 4> definingOps;
      for(auto value : lower()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      for(auto value : lowerext()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      return definingOps;  
    }
    SmallVector<Operation*, 4> getUpperDefiningOps() {
      SmallVector<Operation*, 4> definingOps;
      for(auto value : upper()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      for(auto value : upperext()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      return definingOps;  
    }
    Optional<unsigned> getLowerOperandNumber(unsigned resultNumber) {
      if(resultNumber < lower().size()) {
        return resultNumber;
      }
      return llvm::None;
    }
    Optional<unsigned> getLowerExtraOperandNumber(unsigned resultNumber) {
      if(resultNumber >= lower().size() && resultNumber < lower().size() + lowerext().size()) {
        return resultNumber - lower().size();
      }
      return llvm::None;
    }
    Optional<unsigned> getUpperExtraOperandNumber(unsigned resultNumber) {
      if(resultNumber >= lower().size() + lowerext().size() && resultNumber < lower().size() + lowerext().size() + upperext().size()) {
        return resultNumber - lower().size() - lowerext().size();
      }
      return llvm::None;
    }
    bool isLowerOperand(unsigned operandNumber) {
      return operandNumber < lower().size() ||
        (operandNumber >= lower().size() + upper().size() && 
         operandNumber < lower().size() + upper().size() + lowerext().size());
    }
    bool isUpperOperand(unsigned operandNumber) {
      return !isLowerOperand(operandNumber) && operandNumber < getNumOperands();
    }
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Vtc::CombineOp getCombineTreeRoot();
  }];
}

#endif
