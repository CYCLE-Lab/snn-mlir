#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "snn-mlir/Dialect/SNN/SNNOps.h"
#include "snn-mlir/Dialect/SNN/SNNDialect.h"
#include "snn-mlir/Conversion/SNNToLinalgOps/SNNToLinalgOpspasses.h"

#include <memory>

using namespace mlir;
using namespace snn;

// to define a computeIteratorTypesAndIndexingMaps func
static std::tuple<SmallVector<utils::IteratorType>, SmallVector<AffineMap>>
computeIteratorTypesAndIndexingMaps(OpBuilder &builder, int64_t inputRank) {
    SmallVector<utils::IteratorType> iteratorTypes(inputRank, utils::IteratorType::parallel);
    MLIRContext *ctxt = builder.getContext();
    // Create an identity mapping, representing element-wise access.
    auto identityMap = AffineMap::getMultiDimIdentityMap(inputRank, ctxt);
    SmallVector<AffineMap> indexingMaps{identityMap, identityMap, identityMap};
    return std::make_tuple(iteratorTypes, indexingMaps);
}

// To define a conversion pattern for the LIF 
struct lifOpConversion : public OpRewritePattern<snn::lifOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(snn::lifOp op, PatternRewriter &rewriter) const override {
    Value voltage = op.getVoltage();
    Value input = op.getInputs();
    Location loc = op.getLoc();
    float tau = 0.01f;
    float threshold = 1.0f;
    // calculate (voltage = voltage * (1 - tau) + input)
    Type voltageType = voltage.getType();
    float subtau = 1.0f - tau;
    Value subtauValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(subtau));
    Value tauTensor = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(voltageType).getShape(), cast<ShapedType>(voltageType).getElementType());
    Value tauBroadcast = rewriter.create<linalg::FillOp>(loc, subtauValue, tauTensor).getResult(0);
    Value outVoltage = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(voltageType).getShape(), cast<ShapedType>(voltageType).getElementType());
    Value decayedVoltage = rewriter.create<linalg::MulOp>(loc, outVoltage.getType(), ValueRange{voltage, tauBroadcast}, outVoltage).getResult(0);
    Value finalVoltage = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(voltageType).getShape(),cast<ShapedType>(voltageType).getElementType());
    Value addtensor = rewriter.create<linalg::AddOp>(loc, finalVoltage.getType(), ValueRange{input, decayedVoltage}, finalVoltage ).getResult(0);
    Value thresholdemp = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(addtensor.getType()).getShape(), cast<ShapedType>(addtensor.getType()).getElementType());
    Value thresholdValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(threshold));
    Value thresholdTensor = rewriter.create<linalg::FillOp>(loc, thresholdValue, thresholdemp).getResult(0);

    auto inputRank = cast<ShapedType>(voltageType).getRank();
    auto [iteratorTypes, indexingMaps] = computeIteratorTypesAndIndexingMaps(rewriter, inputRank);
    auto cmpType = RankedTensorType::get(dyn_cast<RankedTensorType>(voltageType).getShape(), dyn_cast<RankedTensorType>(voltageType).getElementType()); 
    Value cmpTensoremp = rewriter.create<tensor::EmptyOp>(loc, dyn_cast<RankedTensorType>(voltageType).getShape(), dyn_cast<RankedTensorType>(voltageType).getElementType());

    // build linalg::GenericOp
    auto updataop = rewriter.create<linalg::GenericOp>(
        loc, cmpType, 
        ValueRange{thresholdTensor, addtensor}, 
        cmpTensoremp, 
        indexingMaps, 
        iteratorTypes, 
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value thresholdElem = args[0]; 
            Value voltageElem = args[1]; 
            Value comparisonResult = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, voltageElem, thresholdElem);
            Value floatZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
            Value floatOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
            Value result = b.create<arith::SelectOp>(loc, comparisonResult, floatZero, floatOne);
            b.create<linalg::YieldOp>(loc, result); 
        }
    );
    Value updataTensor = updataop.getResult(0); 
    rewriter.replaceOp(op, updataTensor);
    
    return success();
  }
};

namespace{
  class SNNToLinalgOpsPass : public mlir::PassWrapper<SNNToLinalgOpsPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SNNToLinalgOpsPass)

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, 
                    mlir::scf::SCFDialect,
                    mlir::linalg::LinalgDialect, 
                    mlir::tensor::TensorDialect>();
    }

    StringRef getArgument() const final {
      return "convert-snn-to-linalg";  
    }

    StringRef getDescription() const final {
      return "Convert SNN operations to Linalg operations."; 
    }

    void runOnOperation() final;
  };
}//namespace

void SNNToLinalgOpsPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<snn::SNNDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect,
                         mlir::scf::SCFDialect,
                         mlir::linalg::LinalgDialect,
                         mlir::tensor::TensorDialect>();
  target.addLegalOp<mlir::linalg::FillOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<lifOpConversion>(&getContext());
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> snn::createSNNToLinalgOpsPass() {
  return std::make_unique<SNNToLinalgOpsPass>();
}

static mlir::PassRegistration<SNNToLinalgOpsPass> pass;
