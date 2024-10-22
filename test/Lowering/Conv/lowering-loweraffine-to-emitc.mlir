// RUN: snn-opt --symbol-dce --convert-memref-to-emitc --convert-func-to-emitc --convert-arith-to-emitc --convert-scf-to-emitc --cse %s | FileCheck %s


module {
  // CHECK: emitc.global static const @__constant_1x2x2x1xf32 : !emitc.array<1x2x2x1xf32> = dense<{{.*}}>
  memref.global "private" constant @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32> = dense<[[[[0.434038758], [0.0581980944]], [[-0.130048275], [0.296388686]]]]> {alignment = 64 : i64}
  // CHECK-LABEL: emitc.func @test_conv
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_conv(%arg0: memref<1x1x2x2xf32> {onnx.name = "input"}, %arg1: memref<1x1x1x1xf32> {onnx.name = "output"}) {
    // CHECK: [[V0:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 2 : index}> : () -> index
    // CHECK: [[V1:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
    // CHECK: [[V2:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0.344225883 : f32}> : () -> f32
    // CHECK: [[V3:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK: [[V4:%[0-9a-zA-Z_]*]] = emitc.get_global @__constant_1x2x2x1xf32 : !emitc.array<1x2x2x1xf32>
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.344225883 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32>
    // CHECK: [[V5:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[V3]], [[V3]], [[V3]], [[V3]]] : (!emitc.array<1x1x1x1xf32>, index, index, index, index) -> f32
    // CHECK: emitc.assign [[V2]] : f32 to [[V5]] : f32
    memref.store %cst, %arg1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    // CHECK: [[V6:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[V3]], [[V3]], [[V3]], [[V3]]] : (!emitc.array<1x1x1x1xf32>, index, index, index, index) -> f32
    // CHECK: emitc.assign [[V2]] : f32 to [[V6]] : f32
    memref.store %cst, %arg1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    // CHECK: emitc.for [[ARG2:%[0-9a-zA-Z_]*]] = [[V3]] to [[V0]] step [[V1]] {
    // CHECK:   emitc.for [[ARG3:%[0-9a-zA-Z_]*]] = [[V3]] to [[V0]] step [[V1]] {
    // CHECK:     [[V7:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG0]][[[V3]], [[V3]], [[ARG2]], [[ARG3]]] : (!emitc.array<1x1x2x2xf32>, index, index, index, index) -> f32
    // CHECK:     [[V8:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V7]] : f32 to [[V8]] : f32
    // CHECK:     [[V9:%[0-9a-zA-Z_]*]] = emitc.subscript [[V4]][[[V3]], [[ARG2]], [[ARG3]], [[V3]]] : (!emitc.array<1x2x2x1xf32>, index, index, index, index) -> f32
    // CHECK:     [[V10:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V9]] : f32 to [[V10]] : f32
    // CHECK:     [[V11:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[V3]], [[V3]], [[V3]], [[V3]]] : (!emitc.array<1x1x1x1xf32>, index, index, index, index) -> f32
    // CHECK:     [[V12:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    // CHECK:     emitc.assign [[V11]] : f32 to [[V12]] : f32
    // CHECK:     [[V13:%[0-9a-zA-Z_]*]] = emitc.mul [[V8]], [[V10]] : (f32, f32) -> f32
    // CHECK:     [[V14:%[0-9a-zA-Z_]*]] = emitc.add [[V12]], [[V13]] : (f32, f32) -> f32
    // CHECK:     [[V15:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[V3]], [[V3]], [[V3]], [[V3]]] : (!emitc.array<1x1x1x1xf32>, index, index, index, index) -> f32
    // CHECK:     emitc.assign [[V14]] : f32 to [[V15]] : f32
    // CHECK:   }
    // CHECK: } 
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %1 = memref.load %arg0[%c0, %c0, %arg2, %arg3] : memref<1x1x2x2xf32>
        %2 = memref.load %0[%c0, %arg2, %arg3, %c0] : memref<1x2x2x1xf32>
        %3 = memref.load %arg1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
        %4 = arith.mulf %1, %2 : f32
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %arg1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      }
    }
    // CHECK: emitc.return
    return
  }
}
