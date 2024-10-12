add_mlir_dialect_library(SNN
		snn.cpp
		
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/snn-mlir/Dialect/Linalg

        DEPENDS
        MLIRSNNIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
