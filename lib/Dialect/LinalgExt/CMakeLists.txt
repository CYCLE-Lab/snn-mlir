add_mlir_dialect_library(LinalgExt
		LinalgExt.cpp
		
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/snn-mlir/Dialect/LinalgExt

        DEPENDS
        MLIRLinalgExtIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
