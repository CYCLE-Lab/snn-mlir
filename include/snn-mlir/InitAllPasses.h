//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef SNN_MLIR_INITALLPASSES_H
#define SNN_MLIR_INITALLPASSES_H

#include "snn-mlir/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"

namespace snn {

// Add all the SNN_MLIR passes.
inline void registerAllPasses() {
  snn::registerPasses();
}

}  // namespace snn

#endif  // SNN_MLIR_INITALLPASSES_H
