//===----------------------------------------------------------------------===//
//
// Defines extension ops for linalg dialect.
//
//===----------------------------------------------------------------------===//

namespace mlir {}
using namespace mlir;

#include "LinalgExt/LinalgExt.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "linalg-ext-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace {
//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Return a `memref.dim` or `tensor.dim` for the shape of `v` at `dim`.
OpFoldResult getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));

  return getAsOpFoldResult(TypeSwitch<Type, Value>(v.getType())
                               .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
                                 return builder.create<tensor::DimOp>(loc, v, dim);
                               })
                               .Case<MemRefType>([&](MemRefType t) -> Value {
                                 return builder.create<memref::DimOp>(loc, v, dim);
                               }));
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
Value getSlice(OpBuilder &b, Location loc, Value source, ArrayRef<OpFoldResult> offsets,
               ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes, strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
      })
      .Default([&](Type t) { return nullptr; });
}

void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    ValueRange results, const ValueRange inputOperands, ValueRange outputOperands) {
  for (auto operand : inputOperands) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand, SideEffects::DefaultResource::get());
  }
  for (auto operand : outputOperands) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand, SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand, SideEffects::DefaultResource::get());
  }
}
}  // Anonymous namespace

namespace linalgext {
//===----------------------------------------------------------------------===//
// GlobalAveragePoolingOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalAveragePoolingOp::verify() {
  ShapedType inputType = getInputOperandType();
  ShapedType outputType = getOutputOperandType();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  if (inputShape.size() != outputShape.size() + 2)
    return emitOpError("incompatible output shape size");

  if (outputShape[0] != inputShape[0])
    return emitOpError("incompatible output shape in dimension: 0");
  if (outputShape[1] != inputShape[3])
    return emitOpError("incompatible output shape in dimension: 1");

  return success();
}

/// Return the iteration domain range.
SmallVector<Range> GlobalAveragePoolingOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getInputOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  const Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  const Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getInput();
  for (const auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> GlobalAveragePoolingOp::getLoopIteratorTypes() {
  return SmallVector<utils::IteratorType>{
      utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction,
      utils::IteratorType::parallel};
}

FailureOr<TilingResult>
GlobalAveragePoolingOp::getTiledImplementation(OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes) {
  LLVM_DEBUG(llvm::dbgs() << "\n[GlobalAveragePoolingOp]@getTiledImplementation offsets is:\n");
  for (const auto &offset : offsets) {
    if (auto attr = offset.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Attribute: " << attr << "\n");
    } else if (auto res = offset.dyn_cast<Value>()) {
      LLVM_DEBUG(llvm::dbgs() << "Value: " << res << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown: " << offset << "\n");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[GlobalAveragePoolingOp]@getTiledImplementation sizes is:\n");
  for (const auto &size : sizes) {
    if (auto attr = size.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Attribute: " << attr << "\n");
    } else if (auto res = size.dyn_cast<Value>()) {
      LLVM_DEBUG(llvm::dbgs() << "Value: " << res << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown: " << size << "\n");
    }
  }

  const int64_t inRank = getInputOperandRank();
  const int64_t outRank = getOutputOperandRank();
  const auto zeroAttr = builder.getI64IntegerAttr(0);
  const auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> inStrides(inRank, oneAttr);
  SmallVector<OpFoldResult> outOffsets(outRank, zeroAttr);
  SmallVector<OpFoldResult> outSizes(outRank, oneAttr);
  SmallVector<OpFoldResult> outStrides(outRank, oneAttr);
  SmallVector<Value> tiledOperands;
  for (const auto i : llvm::seq<int64_t>(0, outRank)) {
    if (i == 0) {
      outOffsets[i] = offsets[i];
      outSizes[i] = sizes[i];
    } else if (i == 1) {
      outOffsets[i] = offsets[i + 2];
      outSizes[i] = sizes[i + 2];
    }
  }
  tiledOperands.emplace_back(getSlice(builder, getLoc(), getInput(), offsets, sizes, inStrides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutput(), outOffsets, outSizes, outStrides));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics())
    resultTypes.push_back(tiledOperands[1].getType());
  Operation *tiledOp = mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult GlobalAveragePoolingOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  LLVM_DEBUG(llvm::dbgs() << "\n[GlobalAveragePoolingOp]@getResultTilePosition offsets is:\n");
  for (const auto &offset : offsets) {
    if (auto attr = offset.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Attribute: " << attr << "\n");
    } else if (auto res = offset.dyn_cast<Value>()) {
      LLVM_DEBUG(llvm::dbgs() << "Value: " << res << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown: " << offset << "\n");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[GlobalAveragePoolingOp]@getResultTilePosition sizes is:\n");
  for (const auto &size : sizes) {
    if (auto attr = size.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Attribute: " << attr << "\n");
    } else if (auto res = size.dyn_cast<Value>()) {
      LLVM_DEBUG(llvm::dbgs() << "Value: " << res << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown: " << size << "\n");
    }
  }

  if (resultNumber == 0) {
    const int64_t outRank = getOutputOperandRank();
    const auto zeroAttr = builder.getI64IntegerAttr(0);
    const auto oneAttr = builder.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> outOffsets(outRank, zeroAttr);
    SmallVector<OpFoldResult> outSizes(outRank, oneAttr);
    for (const auto i : llvm::seq<int64_t>(0, outRank)) {
      if (i == 0) {
        outOffsets[i] = offsets[i];
        outSizes[i] = sizes[i];
      } else if (i == 1) {
        outOffsets[i] = offsets[i + 2];
        outSizes[i] = sizes[i + 2];
      }
    }
    resultOffsets.assign(outOffsets.begin(), outOffsets.end());
    resultSizes.assign(outSizes.begin(), outSizes.end());
    return success();
  }
  return failure();
}

/// Return the ArrayAttr of indexing map.
ArrayAttr GlobalAveragePoolingOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  SmallVector<AffineMap> maps;
  maps.push_back(
      llvm::cast<AffineMapAttr>(
          mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[] -> (d0, d1, d2, d3)>", context))
          .getValue());
  maps.push_back(llvm::cast<AffineMapAttr>(
                     mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[] -> (d0, d3)>", context))
                     .getValue());
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

/// Return the indexing map for a `result`.
AffineMap GlobalAveragePoolingOp::getIndexingMapMatchingResultNumber(unsigned resultNumber) {
  assert(resultNumber == 0);

  auto indexingMaps = getIndexingMaps().template getAsValueRange<AffineMapAttr>();
  return *(indexingMaps.begin() + 1);
}

FailureOr<TilingResult>
GlobalAveragePoolingOp::generateResultTileValue(OpBuilder &b, unsigned resultNumber,
                                                ArrayRef<OpFoldResult> offsets,
                                                ArrayRef<OpFoldResult> sizes) {
  LLVM_DEBUG(llvm::dbgs() << "\n[GlobalAveragePoolingOp]@generateResultTileValue offsets is:\n");
  for (const auto &offset : offsets) {
    if (auto attr = offset.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Attribute: " << attr << "\n");
    } else if (auto res = offset.dyn_cast<Value>()) {
      LLVM_DEBUG(llvm::dbgs() << "Value: " << res << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown: " << offset << "\n");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[GlobalAveragePoolingOp]@generateResultTileValue sizes is:\n");
  for (const auto &size : sizes) {
    if (auto attr = size.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Attribute: " << attr << "\n");
    } else if (auto res = size.dyn_cast<Value>()) {
      LLVM_DEBUG(llvm::dbgs() << "Value: " << res << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown: " << size << "\n");
    }
  }

  // Follow the implementation in TilingInterfaceImpl.cpp
  const AffineMap indexingMap = getIndexingMapMatchingResultNumber(resultNumber);
  if (indexingMap.isEmpty()) {
    return emitOpError("unhandled tiled implementation generation when indexingMap is empty");
  }

  const SmallVector<Range> iterationDomain = getIterationDomain(b);
  const SmallVector<utils::IteratorType> iteratorTypes = getLoopIteratorTypes();
  const int64_t inRank = getInputOperandRank();
  const int64_t outRank = getOutputOperandRank();
  SmallVector<OpFoldResult> iterationTileOffsets(inRank), iterationTileSizes(inRank);
  for (const auto &range : llvm::enumerate(iterationDomain)) {
    iterationTileOffsets[range.index()] = range.value().offset;
    iterationTileSizes[range.index()] = range.value().size;
  }

  for (const auto i : llvm::seq<int64_t>(0, inRank)) {
    if (i == 0) {
      iterationTileOffsets[i] = offsets[0];
      iterationTileSizes[i] = sizes[0];
    } else if (i == 3) {
      iterationTileOffsets[i] = offsets[1];
      iterationTileSizes[i] = sizes[1];
    }
  }

  // If there is a gatherSizes attribute, it is handled separately.
  const ::mlir::ArrayAttr gatherSizesAttr = getGatherSizesAttr();
  if (gatherSizesAttr) {
    // delete the gatherSizes attribute.
    auto removedAttr = removeGatherSizesAttr();
    if (removedAttr)
      return emitOpError("remove GatherSizes Attribute Failed!");

    LLVM_DEBUG(llvm::dbgs() << "\n[GlobalAveragePoolingOp]@generateResultTileValue\n"
                            << "  The gatherSizes attribute is : ");
    for (auto size : gatherSizesAttr) {
      LLVM_DEBUG(llvm::dbgs() << size << " ,");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Get the properties of the Gather Op.
    const mlir::Attribute oneAttr = b.getI64IntegerAttr(1);
    mlir::Attribute stepAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> gatherStrides(inRank, oneAttr);
    SmallVector<OpFoldResult> gatherOffsets(inRank), gatherSizes(inRank);
    Block *forBlock = nullptr;
    unsigned offsetDimBias = 0;
    unsigned sizeDimBias = 1;
    // get the Block of for/forall.
    for (const size_t dim : llvm::seq<size_t>(0, inRank)) {
      gatherOffsets[dim] = iterationTileOffsets[dim];
      gatherSizes[dim] = iterationTileSizes[dim];
      if (dim == 0) {
        assert(iteratorTypes[dim] == utils::IteratorType::parallel);
        if (offsets[dim].is<Value>()) {
          Value arg = cast<Value>(offsets[dim]);
          if (mlir::isa<BlockArgument>(arg)) {
            forBlock = dyn_cast<BlockArgument>(arg).getOwner();
          } else if (mlir::isa<mlir::affine::AffineApplyOp>(arg.getDefiningOp())) {
            auto inputArgs =
                mlir::cast<mlir::affine::AffineApplyOp>(arg.getDefiningOp()).getMapOperands();
            assert(inputArgs.size() == 1 && "only support affineMap has one input!");
            forBlock = dyn_cast<BlockArgument>(inputArgs[0]).getOwner();
          } else {
            arg.dump();
            return emitOpError("offsets[0] is not a BlockArgument!");
          }
          // LLVM_DEBUG(llvm::dbgs() << "  The offsets[0] for block is:    ");
          // LLVM_DEBUG(forBlock->dump());
        }
      } else if (dim == 3) {
        if (offsets[dim - 2].is<Value>() && forBlock == nullptr) {
          Value arg = cast<Value>(offsets[dim - 2]);
          if (mlir::isa<BlockArgument>(arg)) {
            forBlock = dyn_cast<BlockArgument>(arg).getOwner();
          } else if (mlir::isa<mlir::affine::AffineApplyOp>(arg.getDefiningOp())) {
            auto inputArgs =
                mlir::cast<mlir::affine::AffineApplyOp>(arg.getDefiningOp()).getMapOperands();
            assert(inputArgs.size() == 1 && "only support affineMap has one input!");
            forBlock = dyn_cast<BlockArgument>(inputArgs[0]).getOwner();
            offsetDimBias = 1;
          } else {
            arg.dump();
            return emitOpError("offsets[1] is not a BlockArgument!");
          }
          // LLVM_DEBUG(llvm::dbgs() << "  The offsets[1] for block is:    ");
          // LLVM_DEBUG(forBlock->dump());
        }
      }
    }
    assert(forBlock != nullptr
           && "get the Block of for/forall Op Failed,"
              "please make sure the parallel dimension(0 or 3) is splitted!");
    const unsigned gatherDimNums = gatherSizesAttr.size();
    for (const size_t dim : llvm::seq<size_t>(1, inRank - 1)) {
      // update 'forBlock' for scf.for loops structure.
      if (gatherDimNums > dim - 1) {
        assert(iteratorTypes[dim] == utils::IteratorType::reduction);
        BlockArgument arg = forBlock->getArgument(dim - offsetDimBias);
        // if arg is not index, which means loop structure is scf.for.
        if (!arg.getType().isIndex()) {
          // traverse the block to find the first scf.for operation.
          for (mlir::Operation &op : forBlock->getOperations()) {
            // check if the operation is `scf.for`.
            if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
              forBlock = forOp.getBody();
              offsetDimBias = dim;
              auto stepValue = forOp.getStep();
              if (auto *defOp = stepValue.getDefiningOp()) {
                if (auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
                  stepAttr = constOp.getValue();
                  LLVM_DEBUG(llvm::dbgs() << "  The for step is:    ");
                  LLVM_DEBUG(stepAttr.dump());
                } else {
                  constOp.dump();
                  return emitOpError("forOp step is not ConstantOp!");
                }
              } else {
                defOp->dump();
                return emitOpError("forOp getDefiningOp Error!");
              }
              break;
            }
          }
          // if scf.for operation is not found, then search the outside of block.
          if (offsetDimBias != dim) {
            Operation *op = nullptr;
            if ((dim == 1 && gatherDimNums == 1) || dim == 2) {
              op = forBlock->getParentOp()->getParentOp();
            } else if (dim == 1 && gatherDimNums == 2) {
              op = forBlock->getParentOp()->getParentOp()->getParentOp();
            } else {
              return emitOpError()
                     << "unsupported structure(dim=" << dim << ", gatherDimNums=" << gatherDimNums
                     << "), when search the outside of block!";
            }
            // check if the operation is `scf.for`.
            if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
              forBlock = forOp.getBody();
              offsetDimBias = dim;
              auto stepValue = forOp.getStep();
              if (auto *defOp = stepValue.getDefiningOp()) {
                if (auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
                  stepAttr = constOp.getValue();
                  LLVM_DEBUG(llvm::dbgs() << "  The for step is:    ");
                  LLVM_DEBUG(stepAttr.dump());
                } else {
                  constOp.dump();
                  return emitOpError("forOp step is not ConstantOp!");
                }
              } else {
                defOp->dump();
                return emitOpError("forOp getDefiningOp Error!");
              }
            } else {
              op->dump();
              return emitOpError("the updated operation is not scf::ForOp!");
            }
          }
          assert(offsetDimBias == dim && "loop structure is not supported!");
          // LLVM_DEBUG(llvm::dbgs() << "  The updated scf.for block is:    ");
          // LLVM_DEBUG(forBlock->dump());
        }
      }
      // set up gatherOffsets and gatherSizes in H dimesnsion.
      if (dim == 1 && gatherDimNums > 0) {
        assert(iteratorTypes[dim] == utils::IteratorType::reduction);
        OpFoldResult gatherOffset;
        mlir::Attribute gatherSize = gatherSizesAttr[dim - sizeDimBias];
        BlockArgument arg = forBlock->getArgument(dim - offsetDimBias);
        if (Value offVal = dyn_cast<Value>(arg)) {
          //  If gather size be equal step size in dim dimension.
          if (dyn_cast<mlir::IntegerAttr>(gatherSize).getInt()
              == dyn_cast<mlir::IntegerAttr>(stepAttr).getInt()) {
            gatherOffset = offVal;
          } else {
            auto expr = b.getAffineDimExpr(0) * dyn_cast<mlir::IntegerAttr>(gatherSize).getInt();
            Value applyOp =
                affine::makeComposedAffineApply(b, getLoc(), AffineMap::get(1, 0, expr), {offVal});
            LLVM_DEBUG(llvm::dbgs() << "  The H dimension offset is:    ");
            LLVM_DEBUG(applyOp.dump());
            gatherOffset = applyOp;
          }
        } else {
          arg.dump();
          return emitOpError("forBlock->getArgument() is not a Value in H dimension!");
        }
        gatherOffsets[dim] = gatherOffset;
        gatherSizes[dim] = gatherSize;
      }
      // set up gatherOffsets and gatherSizes in W dimesnsion.
      else if (dim == 2 && gatherDimNums > 1) {
        assert(iteratorTypes[dim] == utils::IteratorType::reduction);
        OpFoldResult gatherOffset;
        mlir::Attribute gatherSize = gatherSizesAttr[dim - sizeDimBias];
        BlockArgument arg = forBlock->getArgument(dim - offsetDimBias);
        if (Value offVal = dyn_cast<Value>(arg)) {
          //  If gather size be equal step size in dim dimension.
          if (dyn_cast<mlir::IntegerAttr>(gatherSize).getInt()
              == dyn_cast<mlir::IntegerAttr>(stepAttr).getInt()) {
            gatherOffset = offVal;
          } else {
            auto expr = b.getAffineDimExpr(0) * dyn_cast<mlir::IntegerAttr>(gatherSize).getInt();
            Value applyOp =
                affine::makeComposedAffineApply(b, getLoc(), AffineMap::get(1, 0, expr), {offVal});
            LLVM_DEBUG(llvm::dbgs() << "  The W dimension offset is:    ");
            LLVM_DEBUG(applyOp.dump());
            gatherOffset = applyOp;
          }
        } else {
          arg.dump();
          return emitOpError("forBlock->getArgument() is not a Value in W dimension!");
        }
        gatherOffsets[dim] = gatherOffset;
        gatherSizes[dim] = gatherSize;
      }
    }

    // Get the output properties of the Tiled Op.
    const auto zeroAttr = b.getI64IntegerAttr(0);
    SmallVector<OpFoldResult> outTiledStrides(outRank, oneAttr);
    SmallVector<OpFoldResult> outTiledOffsets(outRank, zeroAttr);
    SmallVector<OpFoldResult> outTiledSizes(outRank, oneAttr);
    SmallVector<Value> tiledOperands;
    for (const auto i : llvm::seq<int64_t>(0, outRank)) {
      if (i == 0) {
        outTiledOffsets[i] = iterationTileOffsets[i];
        outTiledSizes[i] = iterationTileSizes[i];
      } else if (i == 1) {
        outTiledOffsets[i] = iterationTileOffsets[i + 2];
        outTiledSizes[i] = iterationTileSizes[i + 2];
      }
    }

    // insert Gather Op first(InsertSlice Op instead for now!)
    Value gatherInput =
        getSlice(b, getLoc(), getInput(), gatherOffsets, gatherSizes, gatherStrides);
    Value emptyTensor =
        getSlice(b, getLoc(), getInput(), iterationTileOffsets, iterationTileSizes, gatherStrides);
    RankedTensorType emptyTensorTy = dyn_cast<RankedTensorType>(emptyTensor.getType());
    llvm::SmallVector<Value> emptyTensorDynSize;
    for (int i = 0; i < emptyTensorTy.getRank(); i++) {
      if (emptyTensorTy.isDynamicDim(i)) {
        emptyTensorDynSize.push_back(b.create<tensor::DimOp>(getLoc(), emptyTensor, i));
      }
    }
    Value gatherOutput = b.create<tensor::EmptyOp>(
        getLoc(), emptyTensorTy.getShape(), emptyTensorTy.getElementType(), emptyTensorDynSize);
    auto gatherAllOp = b.create<tensor::InsertSliceOp>(getLoc(), gatherInput, gatherOutput,
                                                       gatherOffsets, gatherSizes, gatherStrides);
    LLVM_DEBUG(llvm::dbgs() << "  The Gather Op is:\n    " << gatherAllOp);

    // then insert the Tiled Op.
    tiledOperands.emplace_back(gatherAllOp);
    tiledOperands.emplace_back(
        getSlice(b, getLoc(), getOutput(), outTiledOffsets, outTiledSizes, outTiledStrides));
    SmallVector<Type, 4> resultTypes;
    if (hasPureTensorSemantics())
      resultTypes.push_back(tiledOperands[1].getType());
    Operation *tiledOp = mlir::clone(b, getOperation(), resultTypes, tiledOperands);
    LLVM_DEBUG(llvm::dbgs() << "\n  The Tiled Op is:\n    " << *tiledOp << "\n");

    return TilingResult{{gatherAllOp, tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    ;
  }

  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, iterationTileOffsets, iterationTileSizes);
  if (tilingResult->tiledOps.size() != 1)
    return emitOpError("failed to generate tiled implementation");

  return TilingResult{tilingResult->tiledOps,
                      SmallVector<Value>{tilingResult->tiledValues[resultNumber]}};
}

// cast(dynamic) -> static.
LogicalResult GlobalAveragePoolingOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// TODO: It's not clear at the moment when the function is called,
// so it's possible that the logic needs further inprovement.
LogicalResult
GlobalAveragePoolingOp::reifyResultShapes(OpBuilder &b,
                                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  LLVM_DEBUG(llvm::dbgs() << "\n@reifyResultShapes Start >>>\n");

  SmallVector<OpFoldResult> shapes;
  Location loc = getOperation()->getLoc();
  IRRewriter rewriter(b);
  auto inputShapedType = llvm::cast<ShapedType>(getInputOperandType());
  auto outputShapedType = llvm::cast<ShapedType>(getOutputOperandType());
  SmallVector<utils::IteratorType> iteratorTypes = getLoopIteratorTypes();
  for (int64_t dim : llvm::seq<int64_t>(0, getOutputOperandRank())) {
    if (!outputShapedType.isDynamicDim(dim)) {
      // Static dim: Return IntegerAttr.
      if (iteratorTypes[dim] == utils::IteratorType::parallel) {
        shapes.push_back(b.getIndexAttr(inputShapedType.getDimSize(dim)));
      } else {
        shapes.push_back(b.getI64IntegerAttr(1));
      }
    } else {
      // Dynamic dim: Return Value.
      OpFoldResult ofr = mlir::linalg::createOrFoldDimOp(b, loc, getInput(), dim);
      shapes.push_back(getValueOrCreateConstantIndexOp(b, loc, ofr));
    }
  }
  reifiedReturnShapes.emplace_back(std::move(shapes));

  LLVM_DEBUG(llvm::dbgs() << "\n@reifyResultShapes End <<<\n");
  return success();
}

void GlobalAveragePoolingOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  LLVM_DEBUG(llvm::dbgs() << "\n@GlobalAveragePoolingOp::getEffects() \n");
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(), getDpsInits());
}

//===----------------------------------------------------------------------===//
// LinalgExtDialect
//===----------------------------------------------------------------------===//

void LinalgExtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "graph_compiler/Dialect/LinalgExt/LinalgExtOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "graph_compiler/Dialect/LinalgExt/LinalgExtOps.cpp.inc"
      >();
}
}  // namespace linalgext

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#include "graph_compiler/Dialect/LinalgExt/LinalgExtOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "graph_compiler/Dialect/LinalgExt/LinalgExtOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "graph_compiler/Dialect/LinalgExt/LinalgExtOps.cpp.inc"
