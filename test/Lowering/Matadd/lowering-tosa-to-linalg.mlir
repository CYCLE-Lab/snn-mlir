// RUN: snn-opt --split-input-file --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), func.func(tosa-to-tensor), func.func(tosa-to-arith), func.func(tosa-to-linalg))" %s | FileCheck %s


// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_matadd
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
func.func @test_matadd(%arg0: tensor<4x5xf32> {onnx.name = "x"}, %arg1: tensor<4x5xf32> {onnx.name = "y"}) -> (tensor<4x5xf32> {onnx.name = "output"}) {
  // CHECK: [[EMPTY:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<4x5xf32>
  // CHECK: [[GENERIC:%[0-9a-zA-Z_]*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]} ins([[ARG0]], [[ARG1]] : tensor<4x5xf32>, tensor<4x5xf32>) outs([[EMPTY]] : tensor<4x5xf32>) {
  // CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[IN0:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):  
  // CHECK:   [[ADD:%[0-9a-zA-Z_]*]] = arith.addf [[IN]], [[IN0]] : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<4x5xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  // CHECK: return [[GENERIC]] : tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}
