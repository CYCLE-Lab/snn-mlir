// RUN: snn-opt --symbol-dce --convert-memref-to-emitc --convert-func-to-emitc --convert-arith-to-emitc --convert-scf-to-emitc --cse %s | FileCheck %s


module {
// CHECK-LABEL: emitc.func @test_fc
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_fc(%arg0: memref<2x1xf32> {onnx.name = "data"}, %arg1: memref<2x1xf32> {onnx.name = "output"}) {
  // CHECK: [[V0:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
  // CHECK: [[V1:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 2 : index}> : () -> index
  // CHECK: [[V2:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = -0.956952333 : f32}> : () -> f32
  // CHECK: [[V3:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0.615485549 : f32}> : () -> f32
  // CHECK: [[V4:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant -0.956952333 : f32
    %cst_0 = arith.constant 0.615485549 : f32
    %c0 = arith.constant 0 : index
  // CHECK: emitc.for [[INX1:%[0-9a-zA-Z_]*]] = [[V4]] to [[V1]] step [[V0]] {
  // CHECK:   [[V5:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[INX1]], [[V4]]] : (!emitc.array<2x1xf32>, index, index) -> f32
  // CHECK:   emitc.assign [[V2]] : f32 to [[V5]] : f32
  // CHECK: }
    scf.for %arg2 = %c0 to %c2 step %c1 {
      memref.store %cst, %arg1[%arg2, %c0] : memref<2x1xf32>
    }
  // CHECK: emitc.for [[INX2:%[0-9a-zA-Z_]*]] = [[V4]] to [[V1]] step [[V0]] {
  // CHECK:   [[V5:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[INX2]], [[V4]]] : (!emitc.array<2x1xf32>, index, index) -> f32
  // CHECK:   emitc.assign [[V2]] : f32 to [[V5]] : f32
  // CHECK:   [[V6:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG0]][[[INX2]], [[V4]]] : (!emitc.array<2x1xf32>, index, index) -> f32
  // CHECK:   [[V7:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK:   emitc.assign [[V6]] : f32 to [[V7]] : f32
  // CHECK:   [[V8:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[INX2]], [[V4]]] : (!emitc.array<2x1xf32>, index, index) -> f32
  // CHECK:   [[V9:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK:   emitc.assign [[V8]] : f32 to [[V9]] : f32
  // CHECK:   [[V10:%[0-9a-zA-Z_]*]] = emitc.mul [[V7]], [[V3]] : (f32, f32) -> f32
  // CHECK:   [[V11:%[0-9a-zA-Z_]*]] = emitc.add [[V9]], [[V10]] : (f32, f32) -> f32
  // CHECK:   [[V12:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[INX2]], [[V4]]] : (!emitc.array<2x1xf32>, index, index) -> f32
  // CHECK:   emitc.assign [[V11]] : f32 to [[V12]] : f32
  // CHECK: }
    scf.for %arg2 = %c0 to %c2 step %c1 {
      memref.store %cst, %arg1[%arg2, %c0] : memref<2x1xf32>
      %0 = memref.load %arg0[%arg2, %c0] : memref<2x1xf32>
      %1 = memref.load %arg1[%arg2, %c0] : memref<2x1xf32>
      %2 = arith.mulf %0, %cst_0 : f32
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg1[%arg2, %c0] : memref<2x1xf32>
    }
  // CHECK: emitc.return
    return
  }
}

