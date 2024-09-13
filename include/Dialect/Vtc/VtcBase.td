#ifndef Vtc_BASE
#define Vtc_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Vtc Dialect
//===----------------------------------------------------------------------===//

def Vtc_Dialect : Dialect {
    let name = "Vtc";
    let description = [{A Vtc dialect for weather and climate.}];
    let cppNamespace = "::mlir::Vtc";
}

//===----------------------------------------------------------------------===//
// Vtc Types
//===----------------------------------------------------------------------===//

def Vtc_IsTempType : CPred<"$_self.isa<::mlir::Vtc::TempType>()">;
def Vtc_IsFieldType : CPred<"$_self.isa<::mlir::Vtc::FieldType>()">;
def Vtc_IsResultType : CPred<"$_self.isa<::mlir::Vtc::ResultType>()">;

def Vtc_Temp : Type<Vtc_IsTempType, "a temporary value">;
def Vtc_Field : Type<Vtc_IsFieldType, "a input/output array">;
def Vtc_Result : Type<Vtc_IsResultType, "a result">;

def Vtc_Element : AnyTypeOf<[F32, F64]>;

def Vtc_Index : Confined<I64ArrayAttr, [ArrayCount<3>]>;
def Vtc_Loop : Confined<I64ArrayAttr, [ArrayCount<4>]>;

//===----------------------------------------------------------------------===//
// Vtc Operation
//===----------------------------------------------------------------------===//

// Base class for all Vtc ops.
class Vtc_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Vtc_Dialect, mnemonic, traits> {

    // For each Vtc op, the following static functions need to be defined in
    // VtcOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

#endif // Vtc_BASE
