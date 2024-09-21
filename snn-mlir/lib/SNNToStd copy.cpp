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



#include "SNN/SNNOps.h"
#include "SNN/SNNDialect.h"
#include "SNN/SNNPasses.h"
#include <memory>

using namespace mlir;


// 定义 LIF 操作的转换模式
struct lifOpLowering : public OpRewritePattern<snn::lifOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::lifOp op, PatternRewriter &rewriter) const override {
  // 获取输入电压和输入信号
  Value voltage = op.getVoltage();
  Value input = op.getInputs();

  // 常量定义
  Location loc = op.getLoc();
  float tau = 0.01f;
  float threshold = 1.0f;

  // 1. 将标量 tau 转换为张量，以匹配 voltage 的类型
  Value tauConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(tau));
  Value tauTensor = rewriter.create<tensor::SplatOp>(loc, voltage.getType(), tauConst);

  // 计算衰减电压和输入加权和 (voltage = voltage * (1 - tau) + input)
  Value oneMinusTau = rewriter.create<arith::SubFOp>(loc, rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f)), tauConst);
  Value oneMinusTauTensor = rewriter.create<tensor::SplatOp>(loc, voltage.getType(), oneMinusTau);

  // 计算 decayedVoltage
  Value decayedVoltage = rewriter.create<arith::MulFOp>(loc, voltage, oneMinusTauTensor);

  // 将标量 1.0f 转换为与 input 相同类型的张量
  Value oneConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
  Value oneTensor = rewriter.create<tensor::SplatOp>(loc, input.getType(), oneConst);

  // 使用张量进行乘法
  Value inputTerm = rewriter.create<arith::MulFOp>(loc, input, oneTensor);

  // 计算更新后的电压
  Value updatedVoltage = rewriter.create<arith::AddFOp>(loc, decayedVoltage, inputTerm);

  // 2. 检查是否超过阈值 (if updatedVoltage > threshold)
  Value thresholdVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(threshold));
  Value thresholdTensor = rewriter.create<tensor::SplatOp>(loc, updatedVoltage.getType(), thresholdVal);

  // 对张量进行比较
  Value cmp = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, updatedVoltage, thresholdTensor);

  Value trueConstant = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));

  // 3. 如果超过阈值，生成脉冲并重置电压
  rewriter.create<scf::IfOp>(loc, trueConstant, [&](OpBuilder &b, Location loc) {
    b.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    b.create<scf::YieldOp>(loc);
  });

  // 用新的 voltage 替换原始的 LIF 操作
  rewriter.replaceOp(op, updatedVoltage);

  return success();
}


};

namespace{
// class SNNToStdPass
//     : public mlir::PassWrapper<SNNToStdPass,
//                                mlir::OperationPass<mlir::ModuleOp>> {
class SNNToStdPass : public mlir::PassWrapper<SNNToStdPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SNNToStdPass)


  void getDependentDialects(mlir::DialectRegistry &registry) const override {
  registry.insert<mlir::arith::ArithDialect, 
                  mlir::scf::SCFDialect, 
                  mlir::tensor::TensorDialect>();  // 添加 TensorDialect
}



  StringRef getArgument() const final {
    return "convert-snn-to-std";  // 命令行标识符
  }

  StringRef getDescription() const final {
    return "Convert SNN operations to standard operations.";  // 描述
  }

  // void getDependentDialects(mlir::DialectRegistry &registry) const override {
  //   registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  // }

  void runOnOperation() final;
};
}



void SNNToStdPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<snn::SNNDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect,
                         mlir::scf::SCFDialect,
                         mlir::tensor::TensorDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<lifOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> snn::createSNNToStdPass() {
  return std::make_unique<SNNToStdPass>();
}

// 注册 Pass
static mlir::PassRegistration<SNNToStdPass> pass;


