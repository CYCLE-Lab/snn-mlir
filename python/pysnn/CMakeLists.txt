project(pysnn)

# # 如果系统中存在 pybind11 就使用它
# find_package(pybind11 QUIET)

# # 如果 find_package 找不到 pybind11，则使用 FetchContent 下载并安装
# if (NOT pybind11_FOUND)
#   include(FetchContent)
#   FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com/pybind/pybind11.git
#   )
#   FetchContent_MakeAvailable(pybind11)
# endif()

include(AddMLIRPython)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

set(SNN_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Using SNN_MLIR_PYTHON_ROOT_DIR in: ${SNN_MLIR_PYTHON_ROOT_DIR}")

declare_mlir_python_sources(SNN-MLIRPythonSources)
declare_mlir_python_sources(SNN-MLIRPythonSources.Dialects
  ROOT_DIR "${SNN_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT SNN-MLIRPythonSources
)

declare_mlir_python_extension(SNN-MLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${SNN_MLIR_PYTHON_ROOT_DIR}"
  SOURCES
    snn_registerEverything.cpp
)

################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SNN-MLIRPythonSources.Dialects
  ROOT_DIR "${SNN_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE snnOps.td
  SOURCES snn.py
  DIALECT_NAME SNN)

add_mlir_python_modules(SNN-MLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir/"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES SNN-MLIRPythonSources
  )

install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "test_onnx.py"
)