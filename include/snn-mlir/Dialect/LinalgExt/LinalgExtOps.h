//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "snn-mlir/Dialect/LinalgExt/LinalgExtDialect.h"

#define GET_OP_CLASSES
#include "snn-mlir/Dialect/LinalgExt/LinalgExt.h.inc"
