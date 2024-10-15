#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "snn-mlir/InitAllDialects.h"
#include "snn-mlir/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  snn::registerAllDialects(registry);

  mlir::registerAllPasses();
  snn::registerAllPasses();

  // 调用 MLIR opt main 函数并传递自定义的 pass
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "SNN optimizer driver\n", registry));
}
