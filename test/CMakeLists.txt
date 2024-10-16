llvm_canonicalize_cmake_booleans(
  SNNMLIR_ENABLE_BINDINGS_PYTHON
  )

# 配置 LIT 测试
configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

# 定义测试依赖项
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

# 添加 LIT 测试套件
add_lit_testsuite(check-snn-lit
  "Running the SNN-MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SNN_MLIR_TEST_DEPENDS})

set_target_properties(check-snn-lit PROPERTIES FOLDER "Tests")

# 添加所有 LIT 测试
add_lit_testsuites(SNN_MLIR
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SNN_MLIR_TEST_DEPENDS})
