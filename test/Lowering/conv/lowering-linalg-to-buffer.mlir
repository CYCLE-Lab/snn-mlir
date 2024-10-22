// RUN: snn-opt -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --buffer-results-to-out-params="hoist-static-allocs=1" --unroll-copy -cse %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
// CHECK: memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<{{.*}}> {alignment = 64 : i64}
// CHECK: memref.global "private" constant @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32> = dense<{{.*}}> {alignment = 64 : i64}
// CHECK-LABEL: func.func @test_conv
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_conv(%arg0: tensor<1x1x2x2xf32> {onnx.name = "input"}) -> (tensor<1x1x1x1xf32> {onnx.name = "output"}) {
  // CHECK: [[V0:%[0-9a-zA-Z_]*]] = memref.get_global @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32>
  // CHECK: [[V1:%[0-9a-zA-Z_]*]] = memref.get_global @__constant_1xf32 : memref<1xf32>
    %cst = arith.constant dense<[[[[0.434038758], [0.0581980944]], [[-0.130048275], [0.296388686]]]]> : tensor<1x2x2x1xf32>
    %cst_0 = arith.constant dense<0.344225883> : tensor<1xf32>
  // CHECK: [[COLLAPSE:%[0-9a-zA-Z_]*]] = memref.collapse_shape %arg0 {{.*}} : memref<1x1x2x2xf32> into memref<1x2x2xf32>
  // CHECK: [[EXPAND:%[0-9a-zA-Z_]*]] = memref.expand_shape %collapse_shape {{.*}} output_shape [1, 2, 2, 1] : memref<1x2x2xf32> into memref<1x2x2x1xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x1x2x2xf32> into tensor<1x2x2xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1], [2, 3]] output_shape [1, 2, 2, 1] : tensor<1x2x2xf32> into tensor<1x2x2x1xf32>
    %0 = tensor.empty() : tensor<1x1x1x1xf32>
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[V1]] : memref<1xf32>) outs([[ARG1]] : memref<1x1x1x1xf32>) {
  // CHECK:   ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
  // CHECK:     linalg.yield [[IN]] : f32
  // CHECK: }
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<1xf32>) outs(%0 : tensor<1x1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x1x1xf32>
  // CHECK: linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins([[EXPAND]], %0 : memref<1x2x2x1xf32>, memref<1x2x2x1xf32>) outs([[ARG1]] : memref<1x1x1x1xf32>)
    %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%expanded, %cst : tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>) outs(%1 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  // CHECK: return
    return %2 : tensor<1x1x1x1xf32>
  }
}

