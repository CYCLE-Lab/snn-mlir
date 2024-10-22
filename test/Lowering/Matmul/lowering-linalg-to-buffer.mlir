// RUN: snn-opt --one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --buffer-results-to-out-params="hoist-static-allocs=1" --unroll-copy --cse %s | FileCheck %s


module {
  // CHECK-LABEL: func.func @test_matmul
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_matmul(%arg0: tensor<2x3xf32> {onnx.name = "x"}, %arg1: tensor<3x2xf32> {onnx.name = "y"}) -> (tensor<2x2xf32> {onnx.name = "output"}) {
    // CHECK: [[EXPAND:%[0-9a-zA-Z_]*]] = memref.expand_shape [[ARG0]] {{.*}} output_shape [1, 2, 3] : memref<2x3xf32> into memref<1x2x3xf32>
    // CHECK: [[EXPAND0:%[0-9a-zA-Z_]*]] = memref.expand_shape [[ARG1]] {{.*}} output_shape [1, 3, 2] : memref<3x2xf32> into memref<1x3x2xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2]] output_shape [1, 3, 2] : tensor<3x2xf32> into tensor<1x3x2xf32>
    // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
    // CHECK: [[ALLOC:%[0-9a-zA-Z_]*]] = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x2x2xf32>
    // CHECK: linalg.fill ins([[CST]] : f32) outs([[ALLOC]] : memref<1x2x2xf32>)
    // CHECK: linalg.batch_matmul ins([[EXPAND]], [[EXPAND0]] : memref<1x2x3xf32>, memref<1x3x2xf32>) outs([[ALLOC]] : memref<1x2x2xf32>)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %2 = linalg.batch_matmul ins(%expanded, %expanded_0 : tensor<1x2x3xf32>, tensor<1x3x2xf32>) outs(%1 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    // CHECK: [[COLLAPSE:%[0-9a-zA-Z_]*]] = memref.collapse_shape %alloc {{.*}} : memref<1x2x2xf32> into memref<2x2xf32>
    // CHECK: affine.for [[ARG3:%[0-9a-zA-Z_]*]] = 0 to 2 {
    // CHECK:   affine.for [[ARG4:%[0-9a-zA-Z_]*]] = 0 to 2 {
    // CHECK:     [[V0:%[0-9a-zA-Z_]*]] = memref.load [[COLLAPSE]][[[ARG3]], [[ARG4]]] : memref<2x2xf32>
    // CHECK:     memref.store [[V0]], [[ARG2]][[[ARG3]], [[ARG4]]] : memref<2x2xf32>
    // CHECK:   }
    // CHECK: }
    %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<1x2x2xf32> into tensor<2x2xf32>
    // CHECK: return
    return %collapsed : tensor<2x2xf32>
  }
}

