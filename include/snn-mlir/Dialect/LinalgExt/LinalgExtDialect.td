//===----------------------------------------------------------------------===//
//
// Defines extension for linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPH_COMPILER_DIALECT_LINALGEXT_LINALGEXTBASE_TD
#define GRAPH_COMPILER_DIALECT_LINALGEXT_LINALGEXTBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// LinalgExt Dialect definition
//===----------------------------------------------------------------------===//

def LinalgExtDialect : Dialect {
  let name = "linalgext";
  let cppNamespace = "::linalgext";

  let summary = [{
    Linalg Extensions.
  }];

  let description = [{
    A dialect designed for experimenting with non-structured operations that
    cannot be represented efficiently/directly by the Linalg dialect.
  }];
}



#endif // GRAPH_COMPILER_DIALECT_LINALGEXT_LINALGEXTBASE_TD