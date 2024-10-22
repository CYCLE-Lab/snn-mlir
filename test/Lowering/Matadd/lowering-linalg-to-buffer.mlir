// RUN: snn-opt --one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --buffer-results-to-out-params="hoist-static-allocs=1" --cse %s | FileCheck %s


// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // CHECK-LABEL: func.func @test_matadd
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_matadd(%arg0: tensor<4x5xf32> {onnx.name = "x"}, %arg1: tensor<4x5xf32> {onnx.name = "y"}) -> (tensor<4x5xf32> {onnx.name = "output"}) {
    // CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]} ins([[ARG0]], [[ARG1]] : memref<4x5xf32>, memref<4x5xf32>) outs([[ARG2]] : memref<4x5xf32>) {
    // CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[IN0:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
    // CHECK:   [[ADD:%[0-9a-zA-Z_]*]] = arith.addf [[IN]], [[IN0]] : f32
    // CHECK:   linalg.yield [[ADD]] : f32
    // CHECK: }
    %0 = tensor.empty() : tensor<4x5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x5xf32>, tensor<4x5xf32>) outs(%0 : tensor<4x5xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<4x5xf32>
    // CHECK: return
    return %1 : tensor<4x5xf32>
  }
}

