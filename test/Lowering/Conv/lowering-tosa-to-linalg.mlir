// RUN: snn-opt --split-input-file --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), func.func(tosa-to-tensor), func.func(tosa-to-arith), func.func(tosa-to-linalg))" %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @test_conv
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
func.func @test_conv(%arg0: tensor<1x1x2x2xf32> {onnx.name = "input"}) -> (tensor<1x1x1x1xf32> {onnx.name = "output"}) {
  // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant dense<{{.*}}> : tensor<1x2x2x1xf32>
  // CHECK: [[CST0:%[0-9a-zA-Z_]*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
  %0 = "tosa.const"() <{value = dense<[[[[0.434038758], [0.0581980944]], [[-0.130048275], [0.296388686]]]]> : tensor<1x2x2x1xf32>}> : () -> tensor<1x2x2x1xf32>
  %1 = "tosa.const"() <{value = dense<0.344225883> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK: [[COLLAPSED:%[0-9a-zA-Z_]*]] = tensor.collapse_shape [[ARG0]] {{.*}} : tensor<1x1x2x2xf32> into tensor<1x2x2xf32>
  // CHECK: [[EXPANDED:%[0-9a-zA-Z_]*]] = tensor.expand_shape [[COLLAPSED]] {{.*}} output_shape [1, 2, 2, 1] : tensor<1x2x2xf32> into tensor<1x2x2x1xf32>
  // CHECK: [[V0:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<1x1x1x1xf32>
  // CHECK: [[V1:%[0-9a-zA-Z_]*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CST0]] : tensor<1xf32>) outs([[V0]] : tensor<1x1x1x1xf32>) {
  // CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
  // CHECK:   linalg.yield [[IN]] : f32
  // CHECK: } -> tensor<1x1x1x1xf32> 
  %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 2, 1>} : (tensor<1x1x2x2xf32>) -> tensor<1x2x2x1xf32>
  // CHECK: [[V2:%[0-9a-zA-Z_]*]] = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins([[EXPANDED]], [[CST]] : tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>) outs([[V1]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %3 = tosa.conv2d %2, %0, %1 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>, tensor<1xf32>) -> tensor<1x1x1x1xf32>
  // CHECK: return [[V2]] : tensor<1x1x1x1xf32>
  return %3 : tensor<1x1x1x1xf32>
}
