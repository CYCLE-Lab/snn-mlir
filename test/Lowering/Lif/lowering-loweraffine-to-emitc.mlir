// RUN: snn-opt --symbol-dce --convert-memref-to-emitc --convert-func-to-emitc --convert-arith-to-emitc --convert-scf-to-emitc --cse %s | FileCheck %s


module {
  // CHECK-LABEL: emitc.func @test_lif
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    // CHECK: [[V0:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
    // CHECK: [[V1:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 2 : index}> : () -> index
    // CHECK: [[V2:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK: [[V3:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    // CHECK: [[V4:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    // CHECK: [[V5:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 9.900000e-01 : f32}> : () -> f32
    // CHECK: [[V6:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    // CHECK: [[V7:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    // CHECK: [[V8:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
    // CHECK: [[V9:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x1xf32>
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
    // CHECK: emitc.for [[ARG3:%[0-9a-zA-Z_]*]] = [[V2]] to [[V1]] step [[V0]] {
    // CHECK:   emitc.for [[ARG4:%[0-9a-zA-Z_]*]] = [[V2]] to [[V1]] step [[V0]] {
    // CHECK:     [[V10:%[0-9a-zA-Z_]*]] = emitc.subscript [[V8]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     emitc.assign [[V5]] : f32 to [[V10]] : f32
    // CHECK:     [[V11:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG0]][[[ARG3]], [[ARG4]]] : (!emitc.array<2x2xf32>, index, index) -> f32
    // CHECK:     [[V12:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V11]] : f32 to [[V12]] : f32
    // CHECK:     [[V13:%[0-9a-zA-Z_]*]] = emitc.subscript [[V8]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     [[V14:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V13]] : f32 to [[V14]] : f32
    // CHECK:     [[V15:%[0-9a-zA-Z_]*]] = emitc.mul [[V12]], [[V14]] : (f32, f32) -> f32
    // CHECK:     [[V16:%[0-9a-zA-Z_]*]] = emitc.subscript [[V9]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     emitc.assign [[V15]] : f32 to [[V16]] : f32
    // CHECK:     [[V17:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[ARG3]], [[ARG4]]] : (!emitc.array<2x2xf32>, index, index) -> f32
    // CHECK:     [[V18:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V17]] : f32 to [[V18]] : f32
    // CHECK:     [[V19:%[0-9a-zA-Z_]*]] = emitc.subscript [[V9]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     [[V20:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V19]] : f32 to [[V20]] : f32
    // CHECK:     [[V21:%[0-9a-zA-Z_]*]] = emitc.add [[V18]], [[V20]] : (f32, f32) -> f32
    // CHECK:     [[V22:%[0-9a-zA-Z_]*]] = emitc.subscript [[V6]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     emitc.assign [[V21]] : f32 to [[V22]] : f32
    // CHECK:     [[V23:%[0-9a-zA-Z_]*]] = emitc.subscript [[V7]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     emitc.assign [[V4]] : f32 to [[V23]] : f32
    // CHECK:     [[V24:%[0-9a-zA-Z_]*]] = emitc.subscript [[V7]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     [[V25:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V24]] : f32 to [[V25]] : f32
    // CHECK:     [[V26:%[0-9a-zA-Z_]*]] = emitc.subscript [[V6]][[[V2]], [[V2]]] : (!emitc.array<1x1xf32>, index, index) -> f32
    // CHECK:     [[V27:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V26]] : f32 to [[V27]] : f32
    // CHECK:     [[V28:%[0-9a-zA-Z_]*]] = emitc.cmp lt, [[V27]], [[V25]] : (f32, f32) -> i1
    // CHECK:     [[V29:%[0-9a-zA-Z_]*]] = emitc.cmp eq, [[V27]], [[V27]] : (f32, f32) -> i1
    // CHECK:     [[V30:%[0-9a-zA-Z_]*]] = emitc.cmp eq, [[V25]], [[V25]] : (f32, f32) -> i1
    // CHECK:     [[V31:%[0-9a-zA-Z_]*]] = emitc.logical_and [[V29]], [[V30]] : i1, i1
    // CHECK:     [[V32:%[0-9a-zA-Z_]*]] = emitc.logical_and [[V31]], [[V28]] : i1, i1
    // CHECK:     [[V33:%[0-9a-zA-Z_]*]] = emitc.conditional [[V32]], [[V3]], [[V4]] : f32
    // CHECK:     [[V34:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG2]][[[ARG3]], [[ARG4]]] : (!emitc.array<2x2xf32>, index, index) -> f32
    // CHECK:     emitc.assign [[V33]] : f32 to [[V34]] : f32
    // CHECK:   }
    // CHECK: } 
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
    // CHECK: emitc.return
    return
  }
}