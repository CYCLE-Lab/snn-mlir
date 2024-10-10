module {
  emitc.func @test_lif(%arg0: !emitc.array<2x2xf32>, %arg1: !emitc.array<2x2xf32>, %arg2: !emitc.array<2x2xf32>) {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 2 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %4 = "emitc.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %5 = "emitc.constant"() <{value = 9.900000e-01 : f32}> : () -> f32
    %6 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %7 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %8 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    %9 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    emitc.for %arg3 = %2 to %1 step %0 {
      emitc.for %arg4 = %2 to %1 step %0 {
        %10 = emitc.subscript %8[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %5 : f32 to %10 : f32
        %11 = emitc.subscript %arg0[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        %12 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %11 : f32 to %12 : f32
        %13 = emitc.subscript %8[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        %14 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %13 : f32 to %14 : f32
        %15 = emitc.mul %12, %14 : (f32, f32) -> f32
        %16 = emitc.subscript %9[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %15 : f32 to %16 : f32
        %17 = emitc.subscript %arg1[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        %18 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %17 : f32 to %18 : f32
        %19 = emitc.subscript %9[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        %20 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %19 : f32 to %20 : f32
        %21 = emitc.add %18, %20 : (f32, f32) -> f32
        %22 = emitc.subscript %6[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %21 : f32 to %22 : f32
        %23 = emitc.subscript %7[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        emitc.assign %4 : f32 to %23 : f32
        %24 = emitc.subscript %7[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        %25 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %24 : f32 to %25 : f32
        %26 = emitc.subscript %6[%2, %2] : (!emitc.array<1x1xf32>, index, index) -> f32
        %27 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        emitc.assign %26 : f32 to %27 : f32
        %28 = emitc.cmp lt, %27, %25 : (f32, f32) -> i1
        %29 = emitc.cmp eq, %27, %27 : (f32, f32) -> i1
        %30 = emitc.cmp eq, %25, %25 : (f32, f32) -> i1
        %31 = emitc.logical_and %29, %30 : i1, i1
        %32 = emitc.logical_and %31, %28 : i1, i1
        %33 = emitc.conditional %32, %3, %4 : f32
        %34 = emitc.subscript %arg2[%arg3, %arg4] : (!emitc.array<2x2xf32>, index, index) -> f32
        emitc.assign %33 : f32 to %34 : f32
      }
    }
    emitc.return
  }
}

