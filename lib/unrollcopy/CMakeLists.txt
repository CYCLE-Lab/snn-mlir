add_mlir_dialect_library(UnrollCopy
        unrollcopy.cpp
		
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/unrollcopy

        
        

	LINK_LIBS PUBLIC
	MLIRIR
	)
