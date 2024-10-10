module {
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    %alloc = memref.alloc() : memref<1x1xf32>
    %alloc_0 = memref.alloc() : memref<1x1xf32>
    %alloc_1 = memref.alloc() : memref<1x1xf32>
    %alloc_2 = memref.alloc() : memref<1x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 9.900000e-01 : f32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst_4, %alloc_1[0, 0] : memref<1x1xf32>
        %0 = affine.load %arg0[%arg3, %arg4] : memref<2x2xf32>
        %1 = affine.load %alloc_1[0, 0] : memref<1x1xf32>
        %2 = arith.mulf %0, %1 : f32
        affine.store %2, %alloc_2[0, 0] : memref<1x1xf32>
        %3 = affine.load %arg1[%arg3, %arg4] : memref<2x2xf32>
        %4 = affine.load %alloc_2[0, 0] : memref<1x1xf32>
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %alloc[0, 0] : memref<1x1xf32>
        affine.store %cst_3, %alloc_0[0, 0] : memref<1x1xf32>
        %6 = affine.load %alloc_0[0, 0] : memref<1x1xf32>
        %7 = affine.load %alloc[0, 0] : memref<1x1xf32>
        %8 = arith.cmpf olt, %7, %6 : f32
        %9 = arith.select %8, %cst, %cst_3 : f32
        affine.store %9, %arg2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    return
  }
}

