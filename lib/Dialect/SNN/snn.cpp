//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE Laboratory.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "snn-mlir/Dialect/SNN/SNNDialect.h"
#include "snn-mlir/Dialect/SNN/SNNOps.h"

#include "snn-mlir/Dialect/SNN/SNNDialect.cpp.inc"
#define GET_OP_CLASSES
#include "snn-mlir/Dialect/SNN/SNN.cpp.inc"

using namespace mlir;
using namespace snn;

void SNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "snn-mlir/Dialect/SNN/SNN.cpp.inc"
      >();
}
