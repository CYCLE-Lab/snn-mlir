get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

################################################################################
# Declare native Python extension
################################################################################

declare_mlir_python_sources(SNNMLIRBindingsPythonExtension)

declare_mlir_python_extension(SNNMLIRBindingsPythonExtension.Core
  MODULE_NAME _snn
  ADD_TO_PARENT SNNMLIRBindingsPythonExtension
  SOURCES
    SNNModules.cpp
  EMBED_CAPI_LINK_LIBS
    SNNMLIRCAPISNN
    MLIRCAPITransforms
  PRIVATE_LINK_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    LLVMSupport
)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(SNNMLIRBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    snn-mlir/__init__.py
    snn-mlir/dialects/_ods_common.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${SNNMLIR_BINARY_DIR}/lib/Bindings/Python/snn-mlir/dialects)

declare_mlir_python_sources(SNNMLIRBindingsPythonSources.Dialects
  ADD_TO_PARENT SNNMLIRBindingsPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SNNMLIRBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE snn-mlir/dialects/snnOps.td
  SOURCES snn-mlir/dialects/snn.py
  DIALECT_NAME snn
)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(SNNMLIRBindingsPythonCAPI
  INSTALL_COMPONENT SNNMLIRBindingsPythonModules
  INSTALL_DESTINATION python_packages/snn-mlir_core/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${SNNMLIR_PYTHON_PACKAGES_DIR}/snn-mlir_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine
    SNNMLIRBindingsPythonSources
    SNNMLIRBindingsPythonExtension
)

# Bundle the SNNMLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the scalehls/
# folder and be accessible under the unified "import snn..." namespace.
add_mlir_python_modules(SNNMLIRPythonModules
  ROOT_PREFIX "${SNNMLIR_PYTHON_PACKAGES_DIR}/snn-mlir_core/mlir/"
  INSTALL_PREFIX "python_packages/snn-mlir_core/mlir/"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine
    SNNMLIRBindingsPythonSources
    SNNMLIRBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    SNNMLIRBindingsPythonCAPI
)

# Bundle the SNNMLIR python sources into our package.
add_mlir_python_modules(SNNPythonModules
  ROOT_PREFIX "${SNNMLIR_PYTHON_PACKAGES_DIR}/snn-mlir_core"
  INSTALL_PREFIX "python_packages/snn-mlir_core"
  DECLARED_SOURCES
    SNNMLIRBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    SNNMLIRBindingsPythonCAPI
)
