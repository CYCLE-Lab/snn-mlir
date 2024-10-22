// RUN: snn-opt --split-input-file  --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-tile="tile-size=32" --fold-memref-alias-ops --lower-affine  --cse --promote-buffers-to-stack --arith-expand --sccp --canonicalize="top-down" --symbol-dce  --canonicalize %s | FileCheck %s


#map = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: memref.global "private" constant @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32> = dense<{{.*}}> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<0.344225883> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32> = dense<[[[[0.434038758], [0.0581980944]], [[-0.130048275], [0.296388686]]]]> {alignment = 64 : i64}
  // CHECK-LABEL: func.func @test_conv
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_conv(%arg0: memref<1x1x2x2xf32> {onnx.name = "input"}, %arg1: memref<1x1x1x1xf32> {onnx.name = "output"}) {
    // CHECK: [[C2:%[0-9a-zA-Z_]*]] = arith.constant 2 : index
    // CHECK: [[C1:%[0-9a-zA-Z_]*]] = arith.constant 1 : index
    // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.344225883 : f32
    // CHECK: [[C0:%[0-9a-zA-Z_]*]] = arith.constant 0 : index
    // CHECK: [[V0:%[0-9a-zA-Z_]*]] = memref.get_global @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32>
    %0 = memref.get_global @__constant_1x2x2x1xf32 : memref<1x2x2x1xf32>
    %1 = memref.get_global @__constant_1xf32 : memref<1xf32>
    // CHECK: memref.store [[CST]], [[ARG1]][[[C0]], [[C0]], [[C0]], [[C0]]] : memref<1x1x1x1xf32>
    // CHECK: memref.store [[CST]], [[ARG1]][[[C0]], [[C0]], [[C0]], [[C0]]] : memref<1x1x1x1xf32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2], [3]] : memref<1x1x2x2xf32> into memref<1x2x2xf32>
    %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 2, 2, 1] : memref<1x2x2xf32> into memref<1x2x2x1xf32>
    // CHECK: scf.for [[ARG2:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:   scf.for [[ARG3:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:     [[V1:%[0-9a-zA-Z_]*]] = memref.load [[ARG0]][[[C0]], [[C0]], [[ARG2]], [[ARG3]]] : memref<1x1x2x2xf32>
    // CHECK:     [[V2:%[0-9a-zA-Z_]*]] = memref.load [[V0]][[[C0]], [[ARG2]], [[ARG3]], [[C0]]] : memref<1x2x2x1xf32>
    // CHECK:     [[V3:%[0-9a-zA-Z_]*]] = memref.load [[ARG1]][[[C0]], [[C0]], [[C0]], [[C0]]] : memref<1x1x1x1xf32>
    // CHECK:     [[V4:%[0-9a-zA-Z_]*]] = arith.mulf [[V1]], [[V2]] : f32
    // CHECK:     [[V5:%[0-9a-zA-Z_]*]] = arith.addf [[V3]], [[V4]] : f32
    // CHECK:     memref.store [[V5]], [[ARG1]][[[C0]], [[C0]], [[C0]], [[C0]]] : memref<1x1x1x1xf32>
    // CHECK:   }
    // CHECK: }
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : memref<1xf32>) outs(%arg1 : memref<1x1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%expand_shape, %0 : memref<1x2x2x1xf32>, memref<1x2x2x1xf32>) outs(%arg1 : memref<1x1x1x1xf32>)
    // CHECK: return
    return
  }
}

