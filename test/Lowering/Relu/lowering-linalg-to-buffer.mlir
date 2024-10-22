// RUN: snn-opt --one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --buffer-results-to-out-params="hoist-static-allocs=1" --cse %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
// CHECK-LABEL: func.func @test_relu
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_relu(%arg0: tensor<1x3x4x4xf32> {onnx.name = "data"}) -> (tensor<1x3x4x4xf32> {onnx.name = "output"}) {
// CHECK-NEXT:   linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : memref<1x3x4x4xf32>) outs([[ARG1]] : memref<1x3x4x4xf32>) {
    %0 = tensor.empty() : tensor<1x3x4x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x3x4x4xf32>) outs(%0 : tensor<1x3x4x4xf32>) {
// CHECK-NEXT:   ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):    
    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:     [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     [[CST0:%[0-9a-zA-Z_]*]] = arith.constant 3.40282347E+38 : f32
// CHECK-NEXT:     [[V0:%[0-9a-zA-Z_]*]] = arith.minimumf [[IN]], [[CST0]] : f32
// CHECK-NEXT:     [[V1:%[0-9a-zA-Z_]*]] = arith.maximumf [[V0]], [[CST]] : f32
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 3.40282347E+38 : f32
      %2 = arith.minimumf %in, %cst_0 : f32
      %3 = arith.maximumf %2, %cst : f32
// CHECK-NEXT:     linalg.yield [[V1]] : f32
// CHECK-NEXT:   }
      linalg.yield %3 : f32
    } -> tensor<1x3x4x4xf32>
// CHECK-NEXT: return
    return %1 : tensor<1x3x4x4xf32>
  }
}

