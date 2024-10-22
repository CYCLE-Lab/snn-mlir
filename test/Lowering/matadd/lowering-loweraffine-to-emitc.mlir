// RUN: snn-opt --symbol-dce --convert-memref-to-emitc --convert-func-to-emitc --convert-arith-to-emitc --convert-scf-to-emitc -cse %s | FileCheck %s


module {
  // CHECK-LABEL: emitc.func @test_matadd
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_matadd(%arg0: memref<4x5xf32> {onnx.name = "x"}, %arg1: memref<4x5xf32> {onnx.name = "y"}, %arg2: memref<4x5xf32> {onnx.name = "output"}) {
    // CHECK: [[V0:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
    // CHECK: [[V1:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 5 : index}> : () -> index
    // CHECK: [[V2:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 4 : index}> : () -> index
    // CHECK: [[V3:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0 : index}> : () -> index   
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    // CHECK: emitc.for [[ARG3:%[0-9a-zA-Z_]*]] = [[V3]] to [[V2]] step [[V0]] {
    // CHECK:   emitc.for [[ARG4:%[0-9a-zA-Z_]*]] = [[V3]] to [[V1]] step [[V0]] {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c5 step %c1 {
        // CHECK: [[V4:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG0]][[[ARG3]], [[ARG4]]] : (!emitc.array<4x5xf32>, index, index) -> f32
        // CHECK: [[V5:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        // CHECK: emitc.assign [[V4]] : f32 to [[V5]] : f32
        // CHECK: [[V6:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[ARG3]], [[ARG4]]] : (!emitc.array<4x5xf32>, index, index) -> f32
        // CHECK: [[V7:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
        // CHECK: emitc.assign [[V6]] : f32 to [[V7]] : f32
        // CHECK: [[V8:%[0-9a-zA-Z_]*]] = emitc.add [[V5]], [[V7]] : (f32, f32) -> f32
        // CHECK: [[V9:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG2]][[[ARG3]], [[ARG4]]] : (!emitc.array<4x5xf32>, index, index) -> f32
        // CHECK: emitc.assign [[V8]] : f32 to [[V9]] : f32
        %0 = memref.load %arg0[%arg3, %arg4] : memref<4x5xf32>
        %1 = memref.load %arg1[%arg3, %arg4] : memref<4x5xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %arg2[%arg3, %arg4] : memref<4x5xf32>
      // CHECK: }
    // CHECK: }
      }
    }
    // CHECK: emitc.return
    return
  }
}