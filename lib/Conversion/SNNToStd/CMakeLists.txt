add_mlir_dialect_library(
  SNNToStd
  SNNToStd.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/snn-mlir/Conversion/SNNToStd
  LINK_LIBS
  PUBLIC
  MLIRIR)
