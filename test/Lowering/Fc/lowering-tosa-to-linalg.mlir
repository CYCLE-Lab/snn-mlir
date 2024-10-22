// RUN: snn-opt --split-input-file --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), func.func(tosa-to-tensor), func.func(tosa-to-arith), func.func(tosa-to-linalg))" %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_fc
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
func.func @test_fc(%arg0: tensor<2x1xf32> {onnx.name = "data"}) -> (tensor<2x1xf32> {onnx.name = "output"}) {
  // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant dense<0.615485549> : tensor<1x1xf32>
  // CHECK: [[CST0:%[0-9a-zA-Z_]*]] = arith.constant dense<-0.956952333> : tensor<1xf32>
  // CHECK: [[CST1:%[0-9a-zA-Z_]*]] = arith.constant dense<[1, 0]> : tensor<2xi64>
  // CHECK: [[CST2:%[0-9a-zA-Z_]*]] = arith.constant dense<0.615485549> : tensor<1x1xf32>
  %0 = "tosa.const"() <{value = dense<0.615485549> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
  %1 = "tosa.const"() <{value = dense<-0.956952333> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK: [[EMPTY1:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<2x1xf32>
  // CHECK: [[GENERIC:%[0-9a-zA-Z_]*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins([[CST0]] : tensor<1xf32>) outs([[EMPTY1]] : tensor<2x1xf32>) {
  // CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
  // CHECK:   linalg.yield [[IN]] : f32
  // CHECK: } -> tensor<2x1xf32>
  // CHECK: [[MATMUL:%[0-9a-zA-Z_]*]] = linalg.matmul ins([[ARG0]], [[CST2]] : tensor<2x1xf32>, tensor<1x1xf32>) outs([[GENERIC]] : tensor<2x1xf32>) -> tensor<2x1xf32>
  %2 = tosa.fully_connected %arg0, %0, %1 : (tensor<2x1xf32>, tensor<1x1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  // CHECK: return [[MATMUL]] : tensor<2x1xf32>
  return %2 : tensor<2x1xf32>
}








