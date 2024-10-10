module {
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.900000e-01 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst_1, %alloc[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<2x2xf32>
        %1 = affine.load %alloc[%arg3, %arg4] : memref<2x2xf32>
        %2 = arith.mulf %0, %1 : f32
        affine.store %2, %alloc_2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %arg1[%arg3, %arg4] : memref<2x2xf32>
        %1 = affine.load %alloc_2[%arg3, %arg4] : memref<2x2xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_3[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst_0, %alloc_4[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %alloc_4[%arg3, %arg4] : memref<2x2xf32>
        %1 = affine.load %alloc_3[%arg3, %arg4] : memref<2x2xf32>
        %2 = arith.cmpf olt, %1, %0 : f32
        %3 = arith.select %2, %cst, %cst_0 : f32
        affine.store %3, %arg2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    return
  }
}

