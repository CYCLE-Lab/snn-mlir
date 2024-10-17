//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef SNN_MLIR_INITALLDIALECTS_H
#define SNN_MLIR_INITALLDIALECTS_H

#include "snn-mlir/Dialect/SNN/SNNDialect.h"

namespace snn {

// Add all the SNN_MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<snn::SNNDialect>();
}

}  // namespace snn

#endif  // SNN_MLIR_INITALLDIALECTS_H