
#include "snn-mlir-c/SNN.h"
#include "mlir/CAPI/Registration.h"
#include "snn-mlir/Dialect/SNN/SNNDialect.h"

using namespace mlir;
using namespace snn-mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SNN, snn, snn::SNNDialect)
