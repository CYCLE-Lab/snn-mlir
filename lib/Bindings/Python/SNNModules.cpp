#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include <pybind11/pybind11.h>

// 你自定义的Dialect和Ops定义头文件
#include "snn-mlir/Dialect/SNN/SNNOps.h"
#include "snn-mlir/Dialect/SNN/SNNDialect.h"

namespace py = pybind11;

PYBIND11_MODULE(_snn, m) {
  m.doc() = "SNN-MLIR Python Native Extension(Dialects Registration)";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    // Get the MlirContext capsule from PyMlirContext capsule.
    auto wrappedCapsule = pybind11::capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
    MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

    MlirDialectHandle snn = mlirGetDialectHandle__snn__();
    mlirDialectHandleRegisterDialect(snn, context);
    mlirDialectHandleLoadDialect(snn, context);
  });
}