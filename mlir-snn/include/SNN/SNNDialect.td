#ifndef SNN_DIALECT_TD
#define SNN_DIALECT_TD

include "mlir/IR/OpBase.td"

def SNNDialect : Dialect {
  let name = "snn";
  let cppNamespace = "::snn";
  let summary = "SNN Dialect";
}

#endif
