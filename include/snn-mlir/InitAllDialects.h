#ifndef SNNMLIR_INITALLDIALECTS_H
#define SNNMLIR_INITALLDIALECTS_H

#include "snn-mlir/Dialect/SNN/SNNDialect.h"

namespace snn {

// Add all the SNN_MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<snn::SNNDialect>();
}

} // namespace snn

#endif // SNNMLIR_INITALLDIALECTS_H