#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "snn-mlir/Conversion/unrollcopy/unrollcopypasses.h"


using namespace mlir;
using namespace mlir::memref;


struct MemRefCopyUnrollPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto sourceType = cast<MemRefType>(copyOp.getSource().getType());
    auto targetType = cast<MemRefType>(copyOp.getTarget().getType());
    if (!sourceType.hasStaticShape() || !targetType.hasStaticShape())
      return failure();  
    auto sourceShape = sourceType.getShape();

    Location loc = copyOp.getLoc();
    SmallVector<Value, 4> loopIvs;

    for (size_t i = 0; i < sourceShape.size(); ++i) {
      int64_t dimSize = sourceShape[i];  
      auto loop = rewriter.create<affine::AffineForOp>(loc, 0, dimSize, 1);
      loopIvs.push_back(loop.getInductionVar()); 
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    SmallVector<Value, 4> indices(loopIvs.begin(), loopIvs.end()); 
    Value sourceElement = rewriter.create<memref::LoadOp>(loc, copyOp.getSource(), indices);
    rewriter.create<memref::StoreOp>(loc, sourceElement, copyOp.getTarget(), indices);
    rewriter.eraseOp(copyOp);

    return success();
  }
};

namespace {
  class MemrefCopyToLoopUnrollPass : public mlir::PassWrapper<MemrefCopyToLoopUnrollPass, OperationPass<ModuleOp>> {
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
      registry.insert<mlir::arith::ArithDialect, 
                      mlir::scf::SCFDialect, 
                      mlir::memref::MemRefDialect,
                      mlir::affine::AffineDialect,
                      mlir::linalg::LinalgDialect>();
    }
    void runOnOperation() override {
      ConversionTarget target(getContext());
      target.addIllegalOp<memref::CopyOp>();
      target.addLegalDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,  mlir::affine::AffineDialect, mlir::linalg::LinalgDialect>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<MemRefCopyUnrollPattern>(&getContext());

      if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    }

    StringRef getArgument() const final {
    return "unroll-copy";
    }
  };
}// namespace

std::unique_ptr<Pass> snn::createMemrefCopyToLoopUnrollPass() {
  return std::make_unique<MemrefCopyToLoopUnrollPass>();
}

static mlir::PassRegistration<MemrefCopyToLoopUnrollPass> pass;