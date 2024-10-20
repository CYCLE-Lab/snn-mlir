add_mlir_dialect_library(
    EliminateRedundantTensorOps
    EliminateRedundantTensorOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/snn-mlir/Conversion/EliminateRedundantTensorOps
  LINK_LIBS
  PUBLIC
  MLIRIR)
