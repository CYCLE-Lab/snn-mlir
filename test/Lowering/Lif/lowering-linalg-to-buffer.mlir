// RUN: snn-opt --one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --buffer-results-to-out-params="hoist-static-allocs=1" --cse %s | FileCheck %s


// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // CHECK-LABEL: func.func @test_lif
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_lif(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 9.900000e-01 : f32
    // CHECK: [[ALLOC:%[0-9a-zA-Z_]*]] = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    // CHECK: linalg.fill ins([[CST]] : f32) outs([[ALLOC]] : memref<2x2xf32>)
    %cst = arith.constant 9.900000e-01 : f32
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: [[ALLOC0:%[0-9a-zA-Z_]*]] = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    // CHECK: linalg.mul ins([[ARG0]], [[ALLOC]] : memref<2x2xf32>, memref<2x2xf32>) outs([[ALLOC0]] : memref<2x2xf32>)
    %2 = tensor.empty() : tensor<2x2xf32>
    %3 = linalg.mul ins(%arg0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: [[ALLOC1:%[0-9a-zA-Z_]*]] = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    // CHECK: linalg.add ins([[ARG1]], [[ALLOC0]] : memref<2x2xf32>, memref<2x2xf32>) outs([[ALLOC1]] : memref<2x2xf32>)
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.add ins(%arg1, %3 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: [[ALLOC2:%[0-9a-zA-Z_]*]] = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    // CHECK: [[CST3:%[0-9a-zA-Z_]*]] = arith.constant 1.000000e+00 : f32
    // CHECK: linalg.fill ins([[CST3]] : f32) outs([[ALLOC2]] : memref<2x2xf32>)
    %6 = tensor.empty() : tensor<2x2xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %8 = tensor.empty() : tensor<2x2xf32>
    // CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]} ins([[ALLOC2]], [[ALLOC1]] : memref<2x2xf32>, memref<2x2xf32>) outs([[ARG2]] : memref<2x2xf32>) {
    // CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[IN4:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
    // CHECK:   [[V0:%[0-9a-zA-Z_]*]] = arith.cmpf olt, [[IN4]], [[IN]] : f32
    // CHECK:   [[CST5:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
    // CHECK:   [[V1:%[0-9a-zA-Z_]*]] = arith.select [[V0]], [[CST5]], [[CST3]] : f32
    // CHECK:   linalg.yield [[V1]] : f32
    // CHECK: }
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %5 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%8 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %10 = arith.cmpf olt, %in_1, %in : f32
      %cst_2 = arith.constant 0.000000e+00 : f32
      %cst_3 = arith.constant 1.000000e+00 : f32
      %11 = arith.select %10, %cst_2, %cst_3 : f32
      linalg.yield %11 : f32
    } -> tensor<2x2xf32>
    // CHECK: return 
    return %9 : tensor<2x2xf32>
  }
}