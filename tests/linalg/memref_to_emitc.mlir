module {
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<2x2xf32> to !emitc.array<2x2xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<2x2xf32> to !emitc.array<2x2xf32>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<2x2xf32> to !emitc.array<2x2xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.900000e-01 : f32
    %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %4 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %5 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %6 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %7 = emitc.subscript %5[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %cst_1 : f32 to %7 : f32
        %8 = emitc.subscript %2[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        %9 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %8 : f32 to %9 : f32
        %10 = emitc.subscript %5[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %11 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %10 : f32 to %11 : f32
        %12 = arith.mulf %9, %11 : f32
        %13 = emitc.subscript %6[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %12 : f32 to %13 : f32
        %14 = emitc.subscript %1[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        %15 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %14 : f32 to %15 : f32
        %16 = emitc.subscript %6[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %17 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %16 : f32 to %17 : f32
        %18 = arith.addf %15, %17 : f32
        %19 = emitc.subscript %3[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %18 : f32 to %19 : f32
        %20 = emitc.subscript %4[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %cst_0 : f32 to %20 : f32
        %21 = emitc.subscript %4[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %22 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %21 : f32 to %22 : f32
        %23 = emitc.subscript %3[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %24 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %23 : f32 to %24 : f32
        %25 = arith.cmpf olt, %24, %22 : f32
        %26 = arith.select %25, %cst, %cst_0 : f32
        %27 = emitc.subscript %0[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        emitc.assign %26 : f32 to %27 : f32
      }
    }
    return
  }
}

