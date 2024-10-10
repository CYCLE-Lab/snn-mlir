module {
  emitc.func @test_lif(%arg0: !emitc.array<2x2xf32>, %arg1: !emitc.array<2x2xf32>, %arg2: !emitc.array<2x2xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.900000e-01 : f32
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %4 = emitc.subscript %2[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %cst_1 : f32 to %4 : f32
        %5 = emitc.subscript %arg0[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        %6 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %5 : f32 to %6 : f32
        %7 = emitc.subscript %2[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %8 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %7 : f32 to %8 : f32
        %9 = arith.mulf %6, %8 : f32
        %10 = emitc.subscript %3[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %9 : f32 to %10 : f32
        %11 = emitc.subscript %arg1[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        %12 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %11 : f32 to %12 : f32
        %13 = emitc.subscript %3[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %14 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %13 : f32 to %14 : f32
        %15 = arith.addf %12, %14 : f32
        %16 = emitc.subscript %0[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %15 : f32 to %16 : f32
        %17 = emitc.subscript %1[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %cst_0 : f32 to %17 : f32
        %18 = emitc.subscript %1[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %19 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %18 : f32 to %19 : f32
        %20 = emitc.subscript %0[%c0, %c0] : (!emitc.array<1x1xf32>, index, index) -> f32
        %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %20 : f32 to %21 : f32
        %22 = arith.cmpf olt, %21, %19 : f32
        %23 = arith.select %22, %cst, %cst_0 : f32
        %24 = emitc.subscript %arg2[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        emitc.assign %23 : f32 to %24 : f32
      }
    }
    emitc.return
  }
}

