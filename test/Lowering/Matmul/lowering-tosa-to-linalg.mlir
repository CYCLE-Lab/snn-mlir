// RUN: snn-opt --split-input-file --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), func.func(tosa-to-tensor), func.func(tosa-to-arith), func.func(tosa-to-linalg))" %s | FileCheck %s


// CHECK-LABEL: func.func @test_matmul
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
func.func @test_matmul(%arg0: tensor<2x3xf32> {onnx.name = "x"}, %arg1: tensor<3x2xf32> {onnx.name = "y"}) -> (tensor<2x2xf32> {onnx.name = "output"}) {
  // CHECK: [[EXPAND:%[0-9a-zA-Z_]*]] = tensor.expand_shape [[ARG0]] {{.*}} output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
  // CHECK: [[EXPAND0:%[0-9a-zA-Z_]*]] = tensor.expand_shape [[ARG1]] {{.*}} output_shape [1, 3, 2] : tensor<3x2xf32> into tensor<1x3x2xf32>
  %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 3>} : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
  %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 2>} : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
  // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
  // CHECK: [[V0:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<1x2x2xf32>
  // CHECK: [[V1:%[0-9a-zA-Z_]*]] = linalg.fill ins([[CST]] : f32) outs([[V0]] : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  // CHECK: [[V2:%[0-9a-zA-Z_]*]] = linalg.batch_matmul ins([[EXPAND]], [[EXPAND0]] : tensor<1x2x3xf32>, tensor<1x3x2xf32>) outs([[V1]] : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  // CHECK: [[COLLAPSED:%[0-9a-zA-Z_]*]] = tensor.collapse_shape [[V2]] {{.*}} : tensor<1x2x2xf32> into tensor<2x2xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x2x3xf32>, tensor<1x3x2xf32>) -> tensor<1x2x2xf32>
  %3 = tosa.reshape %2 {new_shape = array<i64: 2, 2>} : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return [[COLLAPSED]] : tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}
