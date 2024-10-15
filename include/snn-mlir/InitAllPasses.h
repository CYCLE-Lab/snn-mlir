#ifndef SNNMLIR_INITALLPASSES_H
#define SNNMLIR_INITALLPASSES_H

#include "snn-mlir/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"

namespace snn {

// Add all the SNN_MLIR passes.
inline void registerAllPasses() {
  snn::registerPasses();
}

} // namespace snn

#endif // SNNMLIR_INITALLPASSES_H
