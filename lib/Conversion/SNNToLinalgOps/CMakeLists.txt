add_mlir_dialect_library(
  SNNToLinalgOps
  SNNToLinalgOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/snn-mlir/Conversion/SNNToLinalgOps
  LINK_LIBS
  PUBLIC
  MLIRIR)
