project(pysnn)

# 如果系统中存在 pybind11 就使用它
find_package(pybind11 QUIET)

# 如果 find_package 找不到 pybind11，则使用 FetchContent 下载并安装
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
  )
  FetchContent_MakeAvailable(pybind11)
endif()

file(GLOB _sources snn_pybind.cpp)

pybind11_add_module(pysnn ${_sources})

target_link_libraries(pysnn PRIVATE 
  MLIRAnalysis
  MLIRIR
  MLIRTransforms
  MLIRParser
  LLVMCore
  LLVMSupport
)

# install(TARGETS pysnn DESTINATION python)