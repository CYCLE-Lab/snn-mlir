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

 

    // 获取 source 和 target 的类型
    auto sourceType = copyOp.getSource().getType().cast<MemRefType>();
    auto targetType = copyOp.getTarget().getType().cast<MemRefType>();

    // 确保是静态维度的 memref
    if (!sourceType.hasStaticShape() || !targetType.hasStaticShape())
      return failure();  
    // 如果有动态维度，直接返回失败，不做转换,因为动态维度要又要涉及到memref.dim这个op

    // 获取 memref 的形状
    auto sourceShape = sourceType.getShape();

    // 创建循环用于元素复制
    Location loc = copyOp.getLoc();
    SmallVector<Value, 4> loopIvs;  // 存储循环的循环索引变量

    // 创建循环 (对于每个维度)
    for (int64_t i = 0; i < sourceShape.size(); ++i) {
      int64_t dimSize = sourceShape[i];  // 获取维度的大小
      // Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);  // 下界为0
      // Value upperBound = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);  // 上界为维度大小
      // Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);  // 步长为1

      // // 创建 For 循环
      // auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

      // 创建 Affine For 循环
      auto loop = rewriter.create<affine::AffineForOp>(loc, 0, dimSize, 1);
      loopIvs.push_back(loop.getInductionVar());  // 存储 induction variable

      // 进入循环体
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // 在最内层循环中插入 load和store 操作，实现 memref.copy 的展开
    SmallVector<Value, 4> indices(loopIvs.begin(), loopIvs.end());  // 使用之前存储的循环变量作为索引

    // 存储归纳变量
    // %i = 0 to N
    // %j = 0 to M
    // loopIvs = [%i, %j]

    // // 创建 indices 向量，用于后续加载和存储操作
    // indices = [%i, %j]
    Value sourceElement = rewriter.create<memref::LoadOp>(loc, copyOp.getSource(), indices);
    rewriter.create<memref::StoreOp>(loc, sourceElement, copyOp.getTarget(), indices);

    // 删除原来的 memref.copy 操作
    rewriter.eraseOp(copyOp);

    return success();
  }
};




namespace {
  class MemrefCopyToLoopUnrollPass
      : public mlir::PassWrapper<MemrefCopyToLoopUnrollPass, OperationPass<ModuleOp>> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
  registry.insert<mlir::arith::ArithDialect, 
                  mlir::scf::SCFDialect, 
                  mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect,
                  mlir::linalg::LinalgDialect>();  // 添加 TensorDialect
}


    void runOnOperation() override {
      // 构建Rewriter，插入逻辑
      ConversionTarget target(getContext());
      target.addIllegalOp<memref::CopyOp>();
      target.addLegalDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,  mlir::affine::AffineDialect, mlir::linalg::LinalgDialect>();

      // 定义重写规则
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<MemRefCopyUnrollPattern>(&getContext());

      // 应用转换
      if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    }

   StringRef getArgument() const final {
    return "unroll-copy";
   }

  };
}  // namespace

// 创建pass
std::unique_ptr<Pass> snn::createMemrefCopyToLoopUnrollPass() {
  return std::make_unique<MemrefCopyToLoopUnrollPass>();
}

//注册pass
static mlir::PassRegistration<MemrefCopyToLoopUnrollPass> pass;