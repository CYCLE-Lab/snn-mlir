add_mlir_dialect_library(SNN
		snn.cpp
        SNNToLinalgOps.cpp
        SNNToStd.cpp
		
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/SNN

        DEPENDS
        MLIRSNNIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
