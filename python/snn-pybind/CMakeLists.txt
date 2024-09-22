project(snn-pybind)

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

pybind11_add_module(snn-pybind snn_pybind.cpp)
target_link_libraries(snn-pybind PUBLIC 
  MLIRTransforms
  MLIRParser
  LLVMCore
  LLVMSupport
)

# install(TARGETS snn-pybind DESTINATION python)