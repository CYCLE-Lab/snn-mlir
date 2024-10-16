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

//定义computeIteratorTypesAndIndexingMaps函数
static std::tuple<SmallVector<utils::IteratorType>, SmallVector<AffineMap>>
computeIteratorTypesAndIndexingMaps(OpBuilder &builder, int64_t inputRank) {
    // 所有维度都并行处理
    SmallVector<utils::IteratorType> iteratorTypes(inputRank, utils::IteratorType::parallel);
    
    MLIRContext *ctxt = builder.getContext();
    // 创建单位映射，表示逐元素访问
    auto identityMap = AffineMap::getMultiDimIdentityMap(inputRank, ctxt);
    
    // 只需要一个映射，表示输入输出的维度直接对应
    SmallVector<AffineMap> indexingMaps{identityMap, identityMap, identityMap};

    return std::make_tuple(iteratorTypes, indexingMaps);
}

// 定义 LIF 操作的转换模式
struct lifOpConversion : public OpRewritePattern<snn::lifOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(snn::lifOp op, PatternRewriter &rewriter) const override {
  // 获取输入电压和输入信号
  Value voltage = op.getVoltage();
  Value input = op.getInputs();

  // 常量定义
  Location loc = op.getLoc();
  float tau = 0.01f;
  float threshold = 1.0f;
  // 计算衰减电压和输入加权和 (voltage = voltage * (1 - tau) + input)
  //计算(1 - tau) 并广播成一个tensor
  Type voltageType = voltage.getType();
  float subtau = 1.0f - tau;
  //auto subtauAttr = rewriter.getF32FloatAttr(subtau);
  Value subtauValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(subtau));
  Value tauTensor = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(voltageType).getShape(), cast<ShapedType>(voltageType).getElementType());
  Value tauBroadcast = rewriter.create<linalg::FillOp>(loc, subtauValue, tauTensor).getResult(0);

  //计算voltage * (1 - tau)
  Value outVoltage = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(voltageType).getShape(), cast<ShapedType>(voltageType).getElementType());
  Value decayedVoltage = rewriter.create<linalg::MulOp>(loc, outVoltage.getType(), ValueRange{voltage, tauBroadcast}, outVoltage).getResult(0);

  //计算voltage * (1 - tau) + input
  Value finalVoltage = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(voltageType).getShape(),cast<ShapedType>(voltageType).getElementType());
  Value addtensor = rewriter.create<linalg::AddOp>(loc, finalVoltage.getType(), ValueRange{input, decayedVoltage}, finalVoltage ).getResult(0);

  //构建阈值tensor
  Value thresholdemp = rewriter.create<tensor::EmptyOp>(loc, cast<ShapedType>(addtensor.getType()).getShape(), cast<ShapedType>(addtensor.getType()).getElementType());
  Value thresholdValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(threshold));
  Value thresholdTensor = rewriter.create<linalg::FillOp>(loc, thresholdValue, thresholdemp).getResult(0);

  // 获取输入张量的类型
  auto inputRank = cast<ShapedType>(voltageType).getRank(); // 输入的维度
  // auto dim = 0; 

  // 使用 computeIteratorTypesAndIndexingMaps 创建迭代器类型和索引映射
  auto [iteratorTypes, indexingMaps] = computeIteratorTypesAndIndexingMaps(rewriter, inputRank);

  // 创建输出张量
  auto cmpType = RankedTensorType::get(dyn_cast<RankedTensorType>(voltageType).getShape(), dyn_cast<RankedTensorType>(voltageType).getElementType()); 
  Value cmpTensoremp = rewriter.create<tensor::EmptyOp>(loc, dyn_cast<RankedTensorType>(voltageType).getShape(), dyn_cast<RankedTensorType>(voltageType).getElementType());

  // 创建 linalg::GenericOp
  auto updataop = rewriter.create<linalg::GenericOp>(
      loc, cmpType, // 输出类型
      ValueRange{thresholdTensor, addtensor}, // 输入张量
      cmpTensoremp, // 输出张量
      indexingMaps, // 使用生成的索引映射
      iteratorTypes, // 使用生成的迭代器类型
      [&](OpBuilder &b, Location loc, ValueRange args) {
          Value thresholdElem = args[0]; // 获取阈值元素
          Value voltageElem = args[1]; // 获取电压元素
          // 进行比较操作，生成布尔值
          Value comparisonResult = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, voltageElem, thresholdElem); // voltageElem小于thresholdElem 则返回true
          // 根据比较结果生成浮点类型的0或1
          Value floatZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
          Value floatOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
          // 选择性地返回0或1
          Value result = b.create<arith::SelectOp>(loc, comparisonResult, floatZero, floatOne);
          b.create<linalg::YieldOp>(loc, result); // 返回结果
      }
  );
  Value updataTensor = updataop.getResult(0); // 获取更新后的张量
  // 用新的 voltage 替换原始的 LIF 操作
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
                  mlir::tensor::TensorDialect>();  // 添加 TensorDialect
}



  StringRef getArgument() const final {
    return "convert-snn-to-linalg";  // 命令行标识符
  }

  StringRef getDescription() const final {
    return "Convert SNN operations to Linalg operations.";  // 描述
  }

  // void getDependentDialects(mlir::DialectRegistry &registry) const override {
  //   registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  // }

  void runOnOperation() final;
};
}



void SNNToLinalgOpsPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<snn::SNNDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect,
                         mlir::scf::SCFDialect,
                         mlir::linalg::LinalgDialect,
                         mlir::tensor::TensorDialect>();
  target.addLegalOp<mlir::linalg::FillOp>(); // 添加 FillOp 到合法操作中

  // 创建转换模式集
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

// 注册 Pass
static mlir::PassRegistration<SNNToLinalgOpsPass> pass;


