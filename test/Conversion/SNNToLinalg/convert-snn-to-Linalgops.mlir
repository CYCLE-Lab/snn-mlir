// RUN: snn-opt -convert-snn-to-linalg %s | FileCheck %s


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_lif
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
func.func @test_lif(%voltage: tensor<2x2xf32>, %input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK-NEXT: [[constant0:[^ ]*]] = arith.constant 9.900000e-01 : f32
  // CHECK-NEXT: [[V0:[^ ]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK-NEXT: [[V1:[^ ]*]] = linalg.fill ins([[constant0]] : f32) outs([[V0]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: [[V2:[^ ]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK-NEXT: [[V3:[^ ]*]] = linalg.mul ins([[ARG0]], [[V1]] : tensor<2x2xf32>, tensor<2x2xf32>) outs([[V2]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: [[V4:[^ ]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK-NEXT: [[V5:[^ ]*]] = linalg.add ins([[ARG1]], [[V3]] : tensor<2x2xf32>, tensor<2x2xf32>) outs([[V4]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: [[V6:[^ ]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK-NEXT: [[constant1:[^ ]*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT: [[V7:[^ ]*]] = linalg.fill ins([[constant1]] : f32) outs([[V6]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: [[V8:[^ ]*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK-NEXT: [[GENERIC:[^ ]*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins([[V7]], [[V5]] : tensor<2x2xf32>, tensor<2x2xf32>) outs([[V8]] : tensor<2x2xf32>) {
  // CHECK-NEXT: ^bb0([[in0:[^ ]*]]: f32, [[in1:[^ ]*]]: f32, [[out:[^ ]*]]: f32):
    // CHECK-NEXT: [[V9:[^ ]*]] = arith.cmpf olt, [[in1]], [[in0]] : f32 
    // CHECK-NEXT: [[constant2:[^ ]*]] = arith.constant 0.000000e+00 : f32
    // CHECK-NEXT: [[constant3:[^ ]*]] = arith.constant 1.000000e+00 : f32
    // CHECK-NEXT: [[V10:[^ ]*]] = arith.select [[V9]], [[constant2]], [[constant3]] : f32
    // CHECK-NEXT: linalg.yield [[V10]] : f32
  // CHECK-NEXT: } -> tensor<2x2xf32>
  %output = "snn.lif"(%voltage, %input) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: return [[GENERIC]] : tensor<2x2xf32>
  return %output : tensor<2x2xf32>
}
