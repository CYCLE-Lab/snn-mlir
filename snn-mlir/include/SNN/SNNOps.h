#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "SNN/SNNDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "SNN/SNN.h.inc"
