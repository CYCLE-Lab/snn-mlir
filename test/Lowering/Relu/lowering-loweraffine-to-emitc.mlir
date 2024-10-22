// RUN: snn-opt --symbol-dce --convert-memref-to-emitc --convert-func-to-emitc --convert-arith-to-emitc --convert-scf-to-emitc --cse %s | FileCheck %s


module {
// CHECK-LABEL: emitc.func @test_relu
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_relu(%arg0: memref<1x3x4x4xf32> {onnx.name = "data"}, %arg1: memref<1x3x4x4xf32> {onnx.name = "output"}) {
  // CHECK-NEXT: [[V0:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 4 : index}> : () -> index
  // CHECK-NEXT: [[V1:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 3 : index}> : () -> index
  // CHECK-NEXT: [[V2:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
  // CHECK-NEXT: [[V3:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
  // CHECK-NEXT: [[V4:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 3.40282347E+38 : f32}> : () -> f32
  // CHECK-NEXT: [[V5:%[0-9a-zA-Z_]*]] = "emitc.constant"() <{value = 0 : index}> : () -> index  
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %c0 = arith.constant 0 : index
  // CHECK-NEXT: emitc.for [[inx1:%[0-9a-zA-Z_]*]] = [[V5]] to [[V1]] step [[V2]] {
  // CHECK-NEXT:   emitc.for [[inx2:%[0-9a-zA-Z_]*]] = [[V5]] to [[V0]] step [[V2]] {
  // CHECK-NEXT:     emitc.for [[inx3:%[0-9a-zA-Z_]*]] = [[V5]] to [[V0]] step [[V2]] {
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
  // CHECK-NEXT:       [[V6:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG0]][[[V5]], [[inx1]], [[inx2]], [[inx3]]] : (!emitc.array<1x3x4x4xf32>, index, index, index, index) -> f32
  // CHECK-NEXT:       [[V7:%[0-9a-zA-Z_]*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK-NEXT:       emitc.assign [[V6]] : f32 to [[V7]] : f32
  // CHECK-NEXT:       [[V8:%[0-9a-zA-Z_]*]] = emitc.cmp lt, [[V7]], [[V4]] : (f32, f32) -> i1
  // CHECK-NEXT:       [[V9:%[0-9a-zA-Z_]*]] = emitc.cmp ne, [[V7]], [[V7]] : (f32, f32) -> i1
  // CHECK-NEXT:       [[V10:%[0-9a-zA-Z_]*]] = emitc.cmp ne, [[V4]], [[V4]] : (f32, f32) -> i1
  // CHECK-NEXT:       [[V11:%[0-9a-zA-Z_]*]] = emitc.logical_or [[V9]], [[V10]] : i1, i1
  // CHECK-NEXT:       [[V12:%[0-9a-zA-Z_]*]] = emitc.logical_or [[V11]], [[V8]] : i1, i1
  // CHECK-NEXT:       [[V13:%[0-9a-zA-Z_]*]] = emitc.conditional [[V12]], [[V7]], [[V4]] : f32  
  // CHECK-NEXT:       [[V14:%[0-9a-zA-Z_]*]] = emitc.cmp gt, [[V13]], [[V3]] : (f32, f32) -> i1 
  // CHECK-NEXT:       [[V15:%[0-9a-zA-Z_]*]] = emitc.cmp ne, [[V13]], [[V13]] : (f32, f32) -> i1 
  // CHECK-NEXT:       [[V16:%[0-9a-zA-Z_]*]] = emitc.cmp ne, [[V3]], [[V3]] : (f32, f32) -> i1 
  // CHECK-NEXT:       [[V17:%[0-9a-zA-Z_]*]] = emitc.logical_or [[V15]], [[V16]] : i1, i1 
  // CHECK-NEXT:       [[V18:%[0-9a-zA-Z_]*]] = emitc.logical_or [[V17]], [[V14]] : i1, i1 
  // CHECK-NEXT:       [[V19:%[0-9a-zA-Z_]*]] = emitc.conditional [[V18]], [[V13]], [[V3]] : f32  
  // CHECK-NEXT:       [[V20:%[0-9a-zA-Z_]*]] = emitc.subscript [[ARG1]][[[V5]], [[inx1]], [[inx2]], [[inx3]]] : (!emitc.array<1x3x4x4xf32>, index, index, index, index) -> f32 
  // CHECK-NEXT:       emitc.assign [[V19]] : f32 to [[V20]] : f32
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }  
          %0 = memref.load %arg0[%c0, %arg2, %arg3, %arg4] : memref<1x3x4x4xf32>
          %1 = arith.cmpf ult, %0, %cst_0 : f32
          %2 = arith.select %1, %0, %cst_0 : f32
          %3 = arith.cmpf ugt, %2, %cst : f32
          %4 = arith.select %3, %2, %cst : f32
          memref.store %4, %arg1[%c0, %arg2, %arg3, %arg4] : memref<1x3x4x4xf32>
        }
      }
    }
  // CHECK-NEXT: emitc.return
    return
  }
}
