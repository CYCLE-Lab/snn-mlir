#include <pybind11/pybind11.h>
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "your_dialect.h" // 你自定义的Dialect和Ops定义头文件

namespace py = pybind11;

PYBIND11_MODULE(your_module_name, m) {
    // 绑定Dialect
    mlir::python::exposeDialectInPython(m, mlir::your_namespace::YourDialect::getDialectNamespace(),
                                        [](mlir::MLIRContext *context) {
        return std::make_unique<mlir::your_namespace::YourDialect>(context);
    });

    // 绑定具体的Op
    mlir::python::exposeOperationInPython<mlir::your_namespace::YourOp>(m, "YourOp");
}
