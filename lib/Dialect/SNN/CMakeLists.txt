add_mlir_dialect_library(MLIRSNN
		snn.cpp
		
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/snn-mlir/Dialect/Linalg

        DEPENDS
        MLIRSNNIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
