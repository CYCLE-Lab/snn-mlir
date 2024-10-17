#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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
#include "snn-mlir/Conversion/SNNToStd/SNNPasses.h"
#include <memory>

using namespace mlir;

// To define a conversion pattern for the LIF
struct lifOpLowering : public OpRewritePattern<snn::lifOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(snn::lifOp op, PatternRewriter &rewriter) const override {
    Value voltage = op.getVoltage();
    Value input = op.getInputs();
    Location loc = op.getLoc();
    float tau = 0.01f;
    float threshold = 1.0f;

    Value tauConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(tau));
    auto tauTensor = rewriter.create<tensor::SplatOp>(loc, voltage.getType(), tauConst);
    Value oneMinusTau = rewriter.create<arith::SubFOp>(
        loc, rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f)), tauTensor);
    Value oneMinusTauTensor = rewriter.create<tensor::SplatOp>(loc, voltage.getType(), oneMinusTau);
    Value decayedVoltage = rewriter.create<arith::MulFOp>(loc, voltage, oneMinusTauTensor);
    Value oneConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    Value oneTensor = rewriter.create<tensor::SplatOp>(loc, input.getType(), oneConst);
    Value inputTerm = rewriter.create<arith::MulFOp>(loc, input, oneTensor);
    Value updatedVoltage = rewriter.create<arith::AddFOp>(loc, decayedVoltage, inputTerm);
    Value thresholdVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(threshold));
    Value thresholdTensor =
        rewriter.create<tensor::SplatOp>(loc, updatedVoltage.getType(), thresholdVal);
    auto tType = cast<RankedTensorType>(updatedVoltage.getType());
    auto shape = tType.getShape();
    Value cmp = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, updatedVoltage,
                                               thresholdTensor);

    // build scf::ForOp
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value dimX = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
    Value dimY = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
    Value OneConsValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    Value ZeroConsValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto outerForOp = rewriter.create<scf::ForOp>(
        loc, zeroIndex, dimX, oneIndex, updatedVoltage,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          Value currentTensor = iterArgs[0];
          auto innerForOp = builder.create<scf::ForOp>(
              loc, zeroIndex, dimY, oneIndex, currentTensor,
              [&](OpBuilder &b, Location loc, Value j, ValueRange innerIterArgs) {
                Value elemCmp = b.create<tensor::ExtractOp>(loc, cmp, ValueRange{i, j});
                auto ifOp = b.create<scf::IfOp>(
                    loc, elemCmp,
                    [&](OpBuilder &b2, Location loc) {
                      Value newTensor = b2.create<tensor::InsertOp>(
                          loc, OneConsValue, innerIterArgs[0], ValueRange{i, j});
                      b2.create<scf::YieldOp>(loc, newTensor);
                    },
                    [&](OpBuilder &b2, Location loc) {
                      Value newTensor = b2.create<tensor::InsertOp>(
                          loc, ZeroConsValue, innerIterArgs[0], ValueRange{i, j});
                      b2.create<scf::YieldOp>(loc, newTensor);
                    });
                b.create<scf::YieldOp>(loc, ifOp->getResult(0));
              });
          builder.create<scf::YieldOp>(loc, innerForOp->getResult(0));
        });
    auto newTensor = outerForOp->getResult(0);
    rewriter.replaceOp(op, newTensor);
    return success();
  }
};

namespace {
class SNNToStdPass : public mlir::PassWrapper<SNNToStdPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SNNToStdPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  StringRef getArgument() const final { return "convert-snn-to-std"; }

  StringRef getDescription() const final {
    return "Convert SNN operations to standard operations.";
  }

  void runOnOperation() final;
};
}  // namespace

void SNNToStdPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<snn::SNNDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                         mlir::tensor::TensorDialect>();
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<lifOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> snn::createSNNToStdPass() { return std::make_unique<SNNToStdPass>(); }

static mlir::PassRegistration<SNNToStdPass> pass;
