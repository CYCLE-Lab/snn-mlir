//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE Laboratory.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef SNN_MLIR_PASSES_H
#define SNN_MLIR_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"

#include "snn-mlir/Conversion/EliminateRedundantTensorOps/EliminateRedundantTensorOpspasses.h"
#include "snn-mlir/Conversion/SNNToLinalgOps/SNNToLinalgOpspasses.h"
#include "snn-mlir/Conversion/SNNToStd/SNNPasses.h"
#include "snn-mlir/Conversion/unrollcopy/unrollcopypasses.h"
#include "snn-mlir/InitAllDialects.h"

namespace snn {

inline void registerPasses() {
  snn::createSNNToStdPass();
  snn::createSNNToLinalgOpsPass();
  snn::createMemrefCopyToLoopUnrollPass();
  snn::createEliminateRedundantTensorOpsPass();
}

}  // namespace snn

#endif  // SNN_MLIR_PASSES_H