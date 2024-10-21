// RUN: snn-opt -split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), func.func(tosa-to-tensor), func.func(tosa-to-arith), func.func(tosa-to-linalg))" %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @test_relu
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  func.func @test_relu(%arg0: tensor<1x3x4x4xf32> {onnx.name = "data"}) -> (tensor<1x3x4x4xf32> {onnx.name = "output"}) {
  // CHECK-NEXT: [[EMPTY:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<1x3x4x4xf32>
  // CHECK-NEXT: [[GENERIC:%[0-9a-zA-Z_]*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x3x4x4xf32>) outs([[EMPTY]] : tensor<1x3x4x4xf32>) {
  // CHECK-NEXT: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
  // CHECK-NEXT:   [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:   [[CST0:%[0-9a-zA-Z_]*]] = arith.constant 3.40282347E+38 : f32
  // CHECK-NEXT:   [[V2:%[0-9a-zA-Z_]*]] = arith.minimumf [[IN]], [[CST0]] : f32
  // CHECK-NEXT:   [[V3:%[0-9a-zA-Z_]*]] = arith.maximumf [[V2]], [[CST]] : f32
    %0 = tosa.clamp %arg0 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  // CHECK-NEXT:   linalg.yield [[V3]] : f32
  // CHECK-NEXT: } -> tensor<1x3x4x4xf32>
  // CHECK-NEXT: return [[GENERIC]] : tensor<1x3x4x4xf32>  
    return %0 : tensor<1x3x4x4xf32>
  }



