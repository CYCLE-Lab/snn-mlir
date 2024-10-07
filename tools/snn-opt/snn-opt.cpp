#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "SNN/SNNDialect.h"      // 引入 SNN 方言
#include "SNN/SNNPasses.h"       // 引入 SNN 定义的 pass



int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // 注册 SNN 方言
  registry.insert<snn::SNNDialect>();

  // 注册必要的其他方言，例如 Arith 和 SCF
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  // snn::registerSNNToStdPass();

  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return snn::createSNNToStdPass();
    });

  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return snn::createMemrefCopyToLoopUnrollPass();
    });  

  // 调用 MLIR opt main 函数并传递自定义的 pass
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "SNN optimizer driver\n", registry));
}



