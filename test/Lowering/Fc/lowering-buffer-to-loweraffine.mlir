// RUN: snn-opt --split-input-file --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-tile="tile-size=32" --fold-memref-alias-ops --lower-affine --cse --promote-buffers-to-stack --arith-expand --sccp --canonicalize="top-down" --symbol-dce --canonicalize %s | FileCheck %s


#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_1x1xf32 : memref<1x1xf32> = dense<0.615485549> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<-0.956952333> {alignment = 64 : i64}
// CHECK-LABEL: func.func @test_fc
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_fc(%arg0: memref<2x1xf32> {onnx.name = "data"}, %arg1: memref<2x1xf32> {onnx.name = "output"}) {
// CHECK: [[C1:%[0-9a-zA-Z_]*]] = arith.constant 1 : index
// CHECK: [[C2:%[0-9a-zA-Z_]*]] = arith.constant 2 : index
// CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant -0.956952333 : f32
// CHECK: [[CST0:%[0-9a-zA-Z_]*]] = arith.constant 0.615485549 : f32
// CHECK: [[C0:%[0-9a-zA-Z_]*]] = arith.constant 0 : index
    %0 = memref.get_global @__constant_1xf32 : memref<1xf32>
    %1 = memref.get_global @__constant_1x1xf32 : memref<1x1xf32>
// CHECK: scf.for [[INX1:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
// CHECK:   memref.store [[CST]], [[ARG1]][[[INX1]], [[C0]]] : memref<2x1xf32>
// CHECK: }
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<1xf32>) outs(%arg1 : memref<2x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
// CHECK: scf.for [[INX2:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
// CHECK:   memref.store [[CST]], [[ARG1]][[[INX2]], [[C0]]] : memref<2x1xf32>
// CHECK:   [[V1:%[0-9a-zA-Z_]*]] = memref.load [[ARG0]][[[INX2]], [[C0]]] : memref<2x1xf32>
// CHECK:   [[V2:%[0-9a-zA-Z_]*]] = memref.load [[ARG1]][[[INX2]], [[C0]]] : memref<2x1xf32>
// CHECK:   [[V3:%[0-9a-zA-Z_]*]] = arith.mulf [[V1]], [[CST0]] : f32
// CHECK:   [[V4:%[0-9a-zA-Z_]*]] = arith.addf [[V2]], [[V3]] : f32
// CHECK:   memref.store [[V4]], [[ARG1]][[[INX2]], [[C0]]] : memref<2x1xf32>
// CHECK: }
    linalg.matmul ins(%arg0, %1 : memref<2x1xf32>, memref<1x1xf32>) outs(%arg1 : memref<2x1xf32>)
// CHECK: return
    return
  }
}




