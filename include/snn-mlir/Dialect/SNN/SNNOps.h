#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "snn-mlir/Dialect/SNN/SNNDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "snn-mlir/Dialect/SNN/SNN.h.inc"
