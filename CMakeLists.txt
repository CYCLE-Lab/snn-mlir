cmake_minimum_required(VERSION 3.20.0)
project(snn-mlir LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(STATUS "SNN-MLIR out-of-tree build.")

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  message(STATUS "SNN-MLIR in-tree build.")

  option(SNN_MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)
  if(SNN_MLIR_ENABLE_BINDINGS_PYTHON)
    message(FATAL_ERROR "Python bindings require a out-of-tree build.")
  endif()

  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)

  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
endif()

# SNN_MLIR Dir sets
set(SNN_MLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SNN_MLIR_BINARY_DIR ${PROJECT_BINARY_DIR})
set(SNN_MLIR_INCLUDE_DIR ${SNN_MLIR_BINARY_DIR}/include)
set(SNN_MLIR_TOOLS_DIR ${SNN_MLIR_BINARY_DIR}/bin)
set(SNN_MLIR_PYTHON_PACKAGES_DIR ${SNN_MLIR_BINARY_DIR}/python_packages)
if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
  message(STATUS "Using SNN_MLIR_SOURCE_DIR in: ${SNN_MLIR_SOURCE_DIR}")
  message(STATUS "Using SNN_MLIR_BINARY_DIR in: ${SNN_MLIR_BINARY_DIR}")
  message(STATUS "Using SNN_MLIR_INCLUDE_DIR in: ${SNN_MLIR_INCLUDE_DIR}")
  message(STATUS "Using SNN_MLIR_TOOLS_DIR in: ${SNN_MLIR_TOOLS_DIR}")
  message(
    STATUS
      "Using SNN_MLIR_PYTHON_PACKAGES_DIR in: ${SNN_MLIR_PYTHON_PACKAGES_DIR}")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SNN_MLIR_SOURCE_DIR}/include)
include_directories(${SNN_MLIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# SNN_MLIR Python Bindings
if(SNN_MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "SNN_MLIR Python bindings are enabled.")

  find_package(
    Python3
    COMPONENTS Interpreter Development NumPy
    REQUIRED)
  if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
    message(STATUS "Found Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Found Python libraries: ${Python3_LIBRARIES}")
    message(STATUS "Found Python executable: ${Python3_EXECUTABLE}")
    message(
      STATUS
        "Found numpy v${Python3_NumPy_VERSION}: ${Python3_NumPy_INCLUDE_DIRS}")
  endif()

  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  include_directories(${Python3_NumPy_INCLUDE_DIRS})

  mlir_detect_pybind11_install()
  find_package(pybind11 CONFIG REQUIRED)
  if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
    message(
      STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
    message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                   "suffix = '${PYTHON_MODULE_SUFFIX}', "
                   "extension = '${PYTHON_MODULE_EXTENSION}'")
  endif()
else()
  message(STATUS "SNN_MLIR Python bindings are disabled.")
endif()

# Subdirectory
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

# Lit
set(LIT_ARGS_DEFAULT "-sv")
if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS
    "${LIT_ARGS_DEFAULT}"
    CACHE STRING "Default options for lit")

# Codes Format Plugin
add_custom_target(
  format
  COMMAND find ${CMAKE_SOURCE_DIR} -name "*.cpp" -o -name "*.h" | xargs
          clang-format -i
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Formatting all source files")
add_custom_target(
  format-cmake
  COMMAND find ${CMAKE_SOURCE_DIR} -name 'CMakeLists.txt' -o -name '*.cmake' |
          xargs cmake-format -i
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Formatting all CMake files")
add_custom_target(
  format-python
  COMMAND black ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Formatting all Python files")

add_custom_target(
  format-all
  DEPENDS format format-cmake format-python
  COMMENT "Formatting all files (C++, CMake, Python)")
