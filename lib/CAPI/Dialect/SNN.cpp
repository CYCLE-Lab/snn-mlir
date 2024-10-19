//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE Laboratory.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "snn-mlir-c/SNN.h"
#include "snn-mlir/Dialect/SNN/SNNDialect.h"

#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace snn;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SNN, snn, snn::SNNDialect)
