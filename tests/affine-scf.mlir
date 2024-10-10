module {
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.900000e-01 : f32
    %alloca = memref.alloca() : memref<1x1xf32>
    %alloca_2 = memref.alloca() : memref<1x1xf32>
    %alloca_3 = memref.alloca() : memref<1x1xf32>
    %alloca_4 = memref.alloca() : memref<1x1xf32>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        memref.store %cst_1, %alloca_3[%c0, %c0] : memref<1x1xf32>
        %0 = memref.load %arg0[%arg3, %arg4] : memref<2x2xf32>
        %1 = memref.load %alloca_3[%c0, %c0] : memref<1x1xf32>
        %2 = arith.mulf %0, %1 : f32
        memref.store %2, %alloca_4[%c0, %c0] : memref<1x1xf32>
        %3 = memref.load %arg1[%arg3, %arg4] : memref<2x2xf32>
        %4 = memref.load %alloca_4[%c0, %c0] : memref<1x1xf32>
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %alloca[%c0, %c0] : memref<1x1xf32>
        memref.store %cst_0, %alloca_2[%c0, %c0] : memref<1x1xf32>
        %6 = memref.load %alloca_2[%c0, %c0] : memref<1x1xf32>
        %7 = memref.load %alloca[%c0, %c0] : memref<1x1xf32>
        %8 = arith.cmpf olt, %7, %6 : f32
        %9 = arith.select %8, %cst, %cst_0 : f32
        memref.store %9, %arg2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    return
  }
}

