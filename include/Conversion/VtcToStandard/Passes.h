#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir
{
    class Pass;
    std::unique_ptr<Pass> createConvertVtcToStandardPass();

#define GEN_PASS_CLASSES
#include "Conversion/VtcToStandard/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/VtcToStandard/Passes.h.inc"

}
