llvm_canonicalize_cmake_booleans(
  SNNMLIR_ENABLE_BINDINGS_PYTHON
  )

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)


set(SNN_MLIR_TEST_DEPENDS
  FileCheck count not
  snn-opt
  split-file
  llvm-dis
)

if(SNNMLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND SNNMLIR_TEST_DEPENDS SNNMLIRPythonModules)
  list(APPEND SNNMLIR_TEST_DEPENDS SNNPythonModules)
endif()


add_lit_testsuite(check-snn-lit
  "Running the SNN-MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SNN_MLIR_TEST_DEPENDS})

set_target_properties(check-snn-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(SNN_MLIR
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SNN_MLIR_TEST_DEPENDS})
