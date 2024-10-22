// RUN: snn-opt --convert-snn-to-linalg %s | FileCheck %s


// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_lif
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
func.func @test_lif(%voltage: tensor<2x2xf32>, %input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 9.900000e-01 : f32
  // CHECK: [[V0:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[V1:%[0-9a-zA-Z_]*]] = linalg.fill ins([[CST]] : f32) outs([[V0]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: [[V2:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[V3:%[0-9a-zA-Z_]*]] = linalg.mul ins([[ARG0]], [[V1]] : tensor<2x2xf32>, tensor<2x2xf32>) outs([[V2]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: [[V4:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[V5:%[0-9a-zA-Z_]*]] = linalg.add ins([[ARG1]], [[V3]] : tensor<2x2xf32>, tensor<2x2xf32>) outs([[V4]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: [[V6:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[CST0:%[0-9a-zA-Z_]*]] = arith.constant 1.000000e+00 : f32
  // CHECK: [[V7:%[0-9a-zA-Z_]*]] = linalg.fill ins([[CST0]] : f32) outs([[V6]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: [[V8:%[0-9a-zA-Z_]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[V9:%[0-9a-zA-Z_]*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]} ins([[V7]], [[V5]] : tensor<2x2xf32>, tensor<2x2xf32>) outs([[V8]] : tensor<2x2xf32>) {
  // CHECK: ^bb0([[IN:%[0-9a-zA-Z_]*]]: f32, [[IN1:%[0-9a-zA-Z_]*]]: f32, [[OUT:%[0-9a-zA-Z_]*]]: f32):
  // CHECK:   [[V10:%[0-9a-zA-Z_]*]] = arith.cmpf olt, [[IN1]], [[IN]] : f32
  // CHECK:   [[CST2:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   [[CST3:%[0-9a-zA-Z_]*]] = arith.constant 1.000000e+00 : f32
  // CHECK:   [[V11:%[0-9a-zA-Z_]*]] = arith.select [[V10]], [[CST2]], [[CST3]] : f32
  // CHECK:   linalg.yield [[V11]] : f32
  // CHECK: } -> tensor<2x2xf32>
  %output = "snn.lif"(%voltage, %input) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return [[V9]] : tensor<2x2xf32>
  return %output : tensor<2x2xf32>
}