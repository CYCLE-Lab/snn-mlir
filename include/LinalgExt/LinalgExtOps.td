//===----------------------------------------------------------------------===//
//
// Defines linalg_ext ops.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPH_COMPILER_DIALECT_LINALGEXT_LINALGEXTOPS_TD
#define GRAPH_COMPILER_DIALECT_LINALGEXT_LINALGEXTOPS_TD

include "LinalgExt/LinalgExtDialect.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Base class for linalg_ext dialect ops.
//===----------------------------------------------------------------------===//

class LinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<LinalgExtDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Global_average_pooling op
//===----------------------------------------------------------------------===//

def LinalgExt_GlobalAveragePoolingOp : LinalgExt_Op<"global_average_pooling",
    [DestinationStyleOpInterface,
     PredOpTrait<"input and output have same element type", TCopVTEtIsSameAs<0, 1>>,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = "Performs global average pooling.";
  let description = [{
    linalg_ext.global_average_pooling computes a numerically stable version of
    global average pooling.

    Layout:
      * Input: NHWC.
      * Output: NC.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       OptionalAttr<I64ArrayAttr>:$gather_sizes
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }

    // TODO: This function code needs to be optimized.
    ArrayAttr getIndexingMaps();
    AffineMap getIndexingMapMatchingResultNumber(unsigned resultNumber);
  }];
  let hasVerifier = 1;
}


#endif // GRAPH_COMPILER_DIALECT_LINALGEXT_LINALGEXTOPS_TD