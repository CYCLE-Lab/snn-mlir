get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# ##############################################################################
# Set up Python binding tools
# ##############################################################################

include(AddMLIRPython)

# ##############################################################################
# Declare native Python extension
# ##############################################################################

declare_mlir_python_sources(SNN_MLIRBindingsPythonExtension)

# add option for ignore warning: the address of ‘MlirTypeID
# mlirUnrankedMemRefTypeGetTypeID()’ \ will never be NULL [-Waddress]
add_compile_options(-Wno-address)

declare_mlir_python_extension(
  SNN_MLIRBindingsPythonExtension.Core
  MODULE_NAME
  _snn
  ADD_TO_PARENT
  SNN_MLIRBindingsPythonExtension
  SOURCES
  SNNModules.cpp
  EMBED_CAPI_LINK_LIBS
  SNN_MLIRCAPISNN
  MLIRCAPITransforms
  PRIVATE_LINK_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  LLVMSupport)

# ##############################################################################
# Declare Python sources
# ##############################################################################

declare_mlir_python_sources(
  SNN_MLIRBindingsPythonSources ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" SOURCES
  snn_mlir/__init__.py snn_mlir/dialects/_ods_common.py)

# ##############################################################################
# Declare dialect-specific bindings.
# ##############################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY
     ${SNN_MLIR_BINARY_DIR}/lib/Bindings/Python/snn_mlir/dialects)

declare_mlir_python_sources(SNN_MLIRBindingsPythonSources.Dialects
                            ADD_TO_PARENT SNN_MLIRBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  SNN_MLIRBindingsPythonSources.Dialects
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE
  snn_mlir/dialects/snnOps.td
  SOURCES
  snn_mlir/dialects/snn.py
  DIALECT_NAME
  snn)

# ##############################################################################
# Build composite binaries
# ##############################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(
  SNN_MLIRBindingsPythonCAPI
  INSTALL_COMPONENT
  SNN_MLIRBindingsPythonModules
  INSTALL_DESTINATION
  python_packages/snn_mlir_core/mlir/_mlir_libs
  OUTPUT_DIRECTORY
  "${SNN_MLIR_PYTHON_PACKAGES_DIR}/snn_mlir_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT
  "../../../.."
  DECLARED_SOURCES
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
  SNN_MLIRBindingsPythonSources
  SNN_MLIRBindingsPythonExtension)

# Bundle the SNN_MLIR python sources into our package. The MLIR API is position
# independent, so we explicitly output it to the mlir/ folder as a temporary
# measure. It will eventually migrate under the scalehls/ folder and be
# accessible under the unified "import snn_mlir..." namespace.
add_mlir_python_modules(
  SNN_MLIRPythonModules
  ROOT_PREFIX
  "${SNN_MLIR_PYTHON_PACKAGES_DIR}/snn_mlir_core/mlir/"
  INSTALL_PREFIX
  "python_packages/snn_mlir_core/mlir/"
  DECLARED_SOURCES
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
  SNN_MLIRBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
  SNN_MLIRBindingsPythonCAPI)

# Bundle the SNN_MLIR python sources into our package.
add_mlir_python_modules(
  SNNPythonModules
  ROOT_PREFIX
  "${SNN_MLIR_PYTHON_PACKAGES_DIR}/snn_mlir_core"
  INSTALL_PREFIX
  "python_packages/snn_mlir_core"
  DECLARED_SOURCES
  SNN_MLIRBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
  SNN_MLIRBindingsPythonCAPI)
