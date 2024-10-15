

# # 设置 LIT 输出目录
# set(LLVM_DEFAULT_EXTERNAL_LIT "")
# set(LLVM_LIT_OUTPUT_DIR ${LLVM_TOOLS_BINARY_DIR})

# # 根据 CMAKE_CFG_INTDIR 设置工具目录
# if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
#   set_llvm_build_mode()
#   string(REPLACE "$<CONFIG>" "${LLVM_BUILD_MODE}" SNN_MLIR_TOOLS_DIR "${SNN_MLIR_RUNTIME_PATH}")
# else()
#   string(REPLACE "$<CONFIG>" "${CMAKE_BUILD_TYPE}" SNN_MLIR_TOOLS_DIR "${SNN_MLIR_RUNTIME_PATH}")
# endif()

# 配置 LIT 测试
configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

# 定义测试依赖项
set(SNN_MLIR_TEST_DEPENDS
  snn-opt
  FileCheck
  count
  not
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
                  DEPENDS
                  ${SNN_MLIR_TEST_DEPENDS})
set_target_properties(check-snn-lit PROPERTIES FOLDER "Tests")

# 添加所有 LIT 测试
add_lit_testsuites(SNN_MLIR
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS
                   ${SNN_MLIR_TEST_DEPENDS})
