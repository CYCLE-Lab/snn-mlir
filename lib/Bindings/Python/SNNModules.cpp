//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include <pybind11/pybind11.h>

#include "snn-mlir-c/SNN.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_snn, m) {
  m.doc() = "SNN-MLIR Python Native Extension(Dialects Registration)";
  auto snn = m.def_submodule("snn");

  snn.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__snn__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
