//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE Laboratory.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "snn-mlir/Dialect/SNN/SNNDialect.h"

#define GET_OP_CLASSES
#include "snn-mlir/Dialect/SNN/SNN.h.inc"
