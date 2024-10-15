################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

set(SNN_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Using SNN_MLIR_PYTHON_ROOT_DIR in: ${SNN_MLIR_PYTHON_ROOT_DIR}")

################################################################################
# Declare native Python extension
################################################################################

declare_mlir_python_extension(SNNMLIRBindingsPythonExtension
  MODULE_NAME _snn
  SOURCES
    SNNModules.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPITransforms
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(SNNMLIRBindingsPythonSources
  ROOT_DIR "${SNN_MLIR_PYTHON_ROOT_DIR}"
  SOURCES
    __init__.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${SNN_MLIR_PYTHON_ROOT_DIR}/dialects)

declare_mlir_python_sources(SNNMLIRBindingsPythonSources.Dialects
  ADD_TO_PARENT SNNMLIRBindingsPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SNNMLIRBindingsPythonSources.Dialects
  ROOT_DIR "${SNN_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/snnOps.td
  SOURCES dialects/snn.py
  DIALECT_NAME snn
)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(SNNMLIRBindingsPythonCAPI
  INSTALL_COMPONENT SNNMLIRPythonModules
  INSTALL_DESTINATION python/mlir/
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SNNMLIRBindingsPythonExtension
)

# Bundle the CIRCT python sources into our package.
add_mlir_python_modules(SNNMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir/"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SNNMLIRBindingsPythonExtension
    SNNMLIRBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    SNNMLIRBindingsPythonCAPI
)
