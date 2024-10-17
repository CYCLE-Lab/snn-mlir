import os
import sys
import lit.util
import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "Super Neural Network"

# Specify the test format.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Define file suffixes that should be treated as test files.
config.suffixes = ['.mlir', '.c', '.cpp', '.py']

# Set the root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Set the root path where tests should be run.
config.test_exec_root = os.path.join(config.snn_mlir_obj_root, "test")

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

config.excludes = [
    'CMakeLists.txt',
    'README.txt',
    'lit.cfg.py'
]

# Adjust the PATH to include the tools directory.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Tweak the PYTHONPATH to include the binary dir.
if config.enable_bindings_python:
  llvm_config.with_environment(
      'PYTHONPATH',
      [os.path.join(config.snn_mlir_python_packages_dir, 'snn_mlir_core')],
      append_path=True)

# Define the tool directories.
tool_dirs = [config.snn_mlir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]

# List of tools to be substituted.
tools = [
    "snn-opt",
    "mlir-opt",
    "mlir-translate",
]

# tools.extend([
#     ToolSubst('%PYTHON', config.python_executable, unresolved='ignore')
# ])

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.enable_bindings_python:
  config.available_features.add('bindings_python')
