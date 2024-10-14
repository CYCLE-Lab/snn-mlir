#include <pybind11/pybind11.h>
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "SNN/SNNDialect.h"
#include "SNN/SNNOps.h" // 你自定义的Dialect和Ops定义头文件

namespace py = pybind11;

PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "SNN-MLIR Dialects Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
}