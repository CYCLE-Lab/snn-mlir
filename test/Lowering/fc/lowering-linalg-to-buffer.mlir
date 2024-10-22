// RUN: snn-opt -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --buffer-results-to-out-params="hoist-static-allocs=1" %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
  #map = affine_map<(d0, d1) -> (0)>
  #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:   memref.global "private" constant @__constant_1x1xf32 : memref<1x1xf32> = dense<0.615485549> {alignment = 64 : i64}
// CHECK:   memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<-0.956952333> {alignment = 64 : i64}
// CHECK-LABEL: func.func @test_fc
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_fc(%arg0: tensor<2x1xf32> {onnx.name = "data"}) -> (tensor<2x1xf32> {onnx.name = "output"}) {
// CHECK: [[V1:%[0-9a-zA-Z_]*]] = memref.get_global @__constant_1xf32 : memref<1xf32>
// CHECK: [[V2:%[0-9a-zA-Z_]*]] = memref.get_global @__constant_1x1xf32 : memref<1x1xf32>
    %cst = arith.constant dense<-0.956952333> : tensor<1xf32>
    %cst_0 = arith.constant dense<0.615485549> : tensor<1x1xf32>
    %0 = tensor.empty() : tensor<2x1xf32>
// CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins([[V1]] : memref<1xf32>) outs([[ARG1]] : memref<2x1xf32>) {
// CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
// CHECK:   linalg.yield [[IN]] : f32
// CHECK: }
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<1xf32>) outs(%0 : tensor<2x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x1xf32>
// CHECK: linalg.matmul ins([[ARG0]], [[V2]] : memref<2x1xf32>, memref<1x1xf32>) outs([[ARG1]] : memref<2x1xf32>)
    %2 = linalg.matmul ins(%arg0, %cst_0 : tensor<2x1xf32>, tensor<1x1xf32>) outs(%1 : tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK: return
    return %2 : tensor<2x1xf32>
  }
