// RUN: snn-opt --split-input-file --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-tile="tile-size=32" --fold-memref-alias-ops --lower-affine --cse --promote-buffers-to-stack --arith-expand --sccp --canonicalize="top-down" --symbol-dce --canonicalize %s | FileCheck %s


module {
  // CHECK-LABEL: func.func @test_matmul
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_matmul(%arg0: memref<2x3xf32> {onnx.name = "x"}, %arg1: memref<3x2xf32> {onnx.name = "y"}, %arg2: memref<2x2xf32> {onnx.name = "output"}) {
    // CHECK: [[C3:%[0-9a-zA-Z_]*]] = arith.constant 3 : index
    // CHECK: [[CS1:%.*]] = arith.constant -1 : index
    // CHECK: [[C2:%[0-9a-zA-Z_]*]] = arith.constant 2 : index
    // CHECK: [[C1:%[0-9a-zA-Z_]*]] = arith.constant 1 : index
    // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
    // CHECK: [[C0:%[0-9a-zA-Z_]*]] = arith.constant 0 : index
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] output_shape [1, 2, 3] : memref<2x3xf32> into memref<1x2x3xf32>
    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1], [2]] output_shape [1, 3, 2] : memref<3x2xf32> into memref<1x3x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    // CHECK: [[ALLOCA:%[0-9a-zA-Z_]*]] = memref.alloca() {alignment = 64 : i64} : memref<1x2x2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf32>
    // CHECK: scf.for [[ARG3:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:   scf.for [[ARG4:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:     memref.store [[CST]], [[ALLOCA]][[[C0]], [[ARG3]], [[ARG4]]] : memref<1x2x2xf32>
    // CHECK:     scf.for [[ARG5:%[0-9a-zA-Z_]*]] = [[C0]] to [[C3]] step [[C1]] {
    // CHECK:       [[V0:%[0-9a-zA-Z_]*]] = memref.load [[ARG0]][[[ARG3]], [[ARG5]]] : memref<2x3xf32>
    // CHECK:       [[V1:%[0-9a-zA-Z_]*]] = memref.load [[ARG1]][[[ARG5]], [[ARG4]]] : memref<3x2xf32>
    // CHECK:       [[V2:%[0-9a-zA-Z_]*]] = memref.load [[ALLOCA]][[[C0]], [[ARG3]], [[ARG4]]] : memref<1x2x2xf32>
    // CHECK:       [[V3:%[0-9a-zA-Z_]*]] = arith.mulf [[V0]], [[V1]] : f32
    // CHECK:       [[V4:%[0-9a-zA-Z_]*]] = arith.addf [[V2]], [[V3]] : f32
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1x2x2xf32>)
    linalg.batch_matmul ins(%expand_shape, %expand_shape_0 : memref<1x2x3xf32>, memref<1x3x2xf32>) outs(%alloc : memref<1x2x2xf32>)
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<1x2x2xf32> into memref<2x2xf32>
    // CHECK: scf.for [[ARG3:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:   scf.for [[ARG4:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:     [[A0:%[0-9a-zA-Z_]*]] = arith.cmpi slt, [[ARG3]], [[C0]] : index
    // CHECK:     [[A1:%[0-9a-zA-Z_]*]] = arith.subi [[CS1]], [[ARG3]] : index
    // CHECK:     [[A2:%[0-9a-zA-Z_]*]] = arith.select [[A0]], [[A1]], [[ARG3]] : index
    // CHECK:     [[A3:%[0-9a-zA-Z_]*]] = arith.divsi [[A2]], [[C2]] : index
    // CHECK:     [[A4:%[0-9a-zA-Z_]*]] = arith.subi [[CS1]], [[A3]] : index
    // CHECK:     [[A5:%[0-9a-zA-Z_]*]] = arith.select [[A0]], [[A4]], [[A3]] : index
    // CHECK:     [[A6:%[0-9a-zA-Z_]*]] = arith.remsi [[ARG3]], [[C2]] : index
    // CHECK:     [[A7:%[0-9a-zA-Z_]*]] = arith.cmpi slt, [[A6]], [[C0]] : index
    // CHECK:     [[A8:%[0-9a-zA-Z_]*]] = arith.addi [[A6]], [[C2]] : index
    // CHECK:     [[A9:%[0-9a-zA-Z_]*]] = arith.select [[A7]], [[A8]], [[A6]] : index
    // CHECK:     [[LOAD:%[0-9a-zA-Z_]*]] = memref.load [[ALLOCA]][[[A5]], [[A9]], [[ARG4]]] : memref<1x2x2xf32>
    // CHECK:     memref.store [[LOAD]], [[ARG2]][[[ARG3]], [[ARG4]]] : memref<2x2xf32>
    // CHECK:   }
    // CHECK: } 
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = memref.load %collapse_shape[%arg3, %arg4] : memref<2x2xf32>
        memref.store %0, %arg2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    // CHECK: return
    return
  }
}



