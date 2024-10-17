#ifndef SNN_MLIR_C_DIALECT_SNN_H
#define SNN_MLIR_C_DIALECT_SNN_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SNN, snn);

#ifdef __cplusplus
}
#endif

#endif  // SNN_MLIR_C_DIALECT_SNN_H