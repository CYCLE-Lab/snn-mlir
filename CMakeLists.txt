cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

project(snn-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(LLVM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm)
find_package(MLIR REQUIRED CONFIG)

if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

set(SNNMLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SNNMLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SNNMLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(SNNMLIR_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(SNNMLIR_PYTHON_PACKAGES_DIR ${SNNMLIR_BINARY_DIR}/python_packages)

if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
  message(STATUS "Using SNNMLIR_SOURCE_DIR in: ${SNNMLIR_SOURCE_DIR}")
  message(STATUS "Using SNNMLIR_BINARY_DIR in: ${SNNMLIR_BINARY_DIR}")
  message(STATUS "Using SNNMLIR_INCLUDE_DIR in: ${SNNMLIR_INCLUDE_DIR}")
  message(STATUS "Using SNNMLIR_TOOLS_DIR in: ${SNNMLIR_TOOLS_DIR}")
  message(STATUS "Using SNNMLIR_PYTHON_PACKAGES_DIR in: ${SNNMLIR_PYTHON_PACKAGES_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

option(SNNMLIR_ENABLE_BINDINGS_PYTHON "Enables SNNMLIR Python bindings." OFF)

if(SNNMLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "SNNMLIR Python bindings are enabled.")

  include(MLIRDetectPythonEnv)
  find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
  if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
    message(STATUS "Found Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Found Python libraries: ${Python3_LIBRARIES}")
    message(STATUS "Found Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "Found numpy v${Python3_NumPy_VERSION}: ${Python3_NumPy_INCLUDE_DIRS}")
  endif()
  include_directories(${Python3_NumPy_INCLUDE_DIRS})

  mlir_detect_pybind11_install()
  find_package(pybind11 CONFIG REQUIRED)
  if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
    message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
    message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                  "suffix = '${PYTHON_MODULE_SUFFIX}', "
                  "extension = '${PYTHON_MODULE_EXTENSION}'")
  endif()
else()
  message(STATUS "SNNMLIR Python bindings are disabled.")
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
