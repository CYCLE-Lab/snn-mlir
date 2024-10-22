// RUN: snn-opt --symbol-dce --convert-memref-to-emitc --convert-func-to-emitc --convert-arith-to-emitc --convert-scf-to-emitc --cse %s | FileCheck %s


module {
  // CHECK-LABEL: emitc.func @test_matmul
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_matmul(%arg0: memref<2x3xf32> {onnx.name = "x"}, %arg1: memref<3x2xf32> {onnx.name = "y"}, %arg2: memref<2x2xf32> {onnx.name = "output"}) {
    // CHECK: [[V0:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 3 : index}> : () -> index
    // CHECK: [[V1:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = -1 : index}> : () -> index
    // CHECK: [[V2:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 2 : index}> : () -> index
    // CHECK: [[V3:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
    // CHECK: [[V4:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    // CHECK: [[V5:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK: [[V6:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1x2x2xf32>
    %c3 = arith.constant 3 : index
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x2x2xf32>
    // CHECK: emitc.for [[ARG3:%[0-9a-zA-Z_]*]] = [[V5]] to [[V2]] step [[V3]] {
    // CHECK:   emitc.for [[ARG4:%[0-9a-zA-Z_]*]] = [[V5]] to [[V2]] step [[V3]] {
    // CHECK:     [[V7:%[0-9a-zA-Z_]*]] = emitc.subscript [[V6]][[[V5]], [[ARG3]], [[ARG4]]] : (!emitc.array<1x2x2xf32>, index, index, index) -> f32
    // CHECK:     emitc.assign [[V4]] : f32 to [[V7]] : f32
    // CHECK:     emitc.for [[ARG5:%[0-9a-zA-Z_]*]] = [[V5]] to [[V0]] step [[V3]] {
    // CHECK:       [[V8:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG0]][[[ARG3]], [[ARG5]]] : (!emitc.array<2x3xf32>, index, index) -> f32
    // CHECK:       [[V9:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:       emitc.assign [[V8]] : f32 to [[V9]] : f32
    // CHECK:       [[V10:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[ARG5]], [[ARG4]]] : (!emitc.array<3x2xf32>, index, index) -> f32
    // CHECK:       [[V11:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:       emitc.assign [[V10]] : f32 to [[V11]] : f32
    // CHECK:       [[V12:%[0-9a-zA-Z_]*]] = emitc.subscript [[V6]][[[V5]], [[ARG3]], [[ARG4]]] : (!emitc.array<1x2x2xf32>, index, index, index) -> f32
    // CHECK:       [[V13:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:       emitc.assign [[V12]] : f32 to [[V13]] : f32
    // CHECK:       [[V14:%[0-9a-zA-Z_]*]] = emitc.mul [[V9]], [[V11]] : (f32, f32) -> f32
    // CHECK:       [[V15:%[0-9a-zA-Z_]*]] = emitc.add [[V13]], [[V14]] : (f32, f32) -> f32
    // CHECK:       [[V16:%[0-9a-zA-Z_]*]] = emitc.subscript [[V6]][[[V5]], [[ARG3]], [[ARG4]]] : (!emitc.array<1x2x2xf32>, index, index, index) -> f32
    // CHECK:       emitc.assign [[V15]] : f32 to [[V16]] : f32
    // CHECK:     }
    // CHECK:   }
    // CHECK: }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        memref.store %cst, %alloca[%c0, %arg3, %arg4] : memref<1x2x2xf32>
        scf.for %arg5 = %c0 to %c3 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<2x3xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<3x2xf32>
          %2 = memref.load %alloca[%c0, %arg3, %arg4] : memref<1x2x2xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %alloca[%c0, %arg3, %arg4] : memref<1x2x2xf32>
        }
      }
    }
    // CHECK: emitc.for [[INX3:%[0-9a-zA-Z_]*]] = [[V5]] to [[V2]] step [[V3]] {
    // CHECK:   emitc.for [[INX4:%[0-9a-zA-Z_]*]] = [[V5]] to [[V2]] step [[V3]] {
    // CHECK:     [[A7:%[0-9a-zA-Z_]*]] = emitc.cmp lt, [[INX3]], [[V5]] : (index, index) -> i1
    // CHECK:     [[A8:%[0-9a-zA-Z_]*]] = emitc.sub [[V1]], [[INX3]] : (index, index) -> index
    // CHECK:     [[A9:%[0-9a-zA-Z_]*]] = emitc.conditional [[A7]], [[A8]], [[INX3]] : index
    // CHECK:     [[A10:%[0-9a-zA-Z_]*]] = emitc.div [[A9]], [[V2]] : (index, index) -> index
    // CHECK:     [[A11:%[0-9a-zA-Z_]*]] = emitc.sub [[V1]], [[A10]] : (index, index) -> index
    // CHECK:     [[A12:%[0-9a-zA-Z_]*]] = emitc.conditional [[A7]], [[A11]], [[A10]] : index
    // CHECK:     [[A13:%[0-9a-zA-Z_]*]] = emitc.rem [[INX3]], [[V2]] : (index, index) -> index
    // CHECK:     [[A14:%[0-9a-zA-Z_]*]] = emitc.cmp lt, [[A13]], [[V5]] : (index, index) -> i1
    // CHECK:     [[A15:%[0-9a-zA-Z_]*]] = emitc.add [[A13]], [[V2]] : (index, index) -> index
    // CHECK:     [[A16:%[0-9a-zA-Z_]*]] = emitc.conditional [[A14]], [[A15]], [[A13]] : index
    // CHECK:     [[A17:%[0-9a-zA-Z_]*]] = emitc.subscript [[V6]][[[A12]], [[A16]], [[INX4]]] : (!emitc.array<1x2x2xf32>, index, index, index) -> f32
    // CHECK:     [[A18:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[A17]] : f32 to [[A18]] : f32
    // CHECK:     [[A19:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG2]][[[INX3]], [[INX4]]] : (!emitc.array<2x2xf32>, index, index) -> f32
    // CHECK:     emitc.assign [[A18]] : f32 to [[A19]] : f32
    // CHECK:   }
    // CHECK: } 
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %0 = arith.cmpi slt, %arg3, %c0 : index
        %1 = arith.subi %c-1, %arg3 : index
        %2 = arith.select %0, %1, %arg3 : index
        %3 = arith.divsi %2, %c2 : index
        %4 = arith.subi %c-1, %3 : index
        %5 = arith.select %0, %4, %3 : index
        %6 = arith.remsi %arg3, %c2 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c2 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = memref.load %alloca[%5, %9, %arg4] : memref<1x2x2xf32>
        memref.store %10, %arg2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    // CHECK: emitc.return
    return
  }
}

