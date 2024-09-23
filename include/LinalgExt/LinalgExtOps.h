#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "LinalgExt/LinalgExtDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "LinalgExt/LinalgExt.h.inc"
