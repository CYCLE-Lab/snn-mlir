// RUN: snn-opt --split-input-file  --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-tile="tile-size=32" --fold-memref-alias-ops --lower-affine  --cse --promote-buffers-to-stack --arith-expand --sccp --canonicalize="top-down" --symbol-dce  --canonicalize %s | FileCheck %s


#map = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
// CHECK-LABEL: func.func @test_relu
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  func.func @test_relu(%arg0: memref<1x3x4x4xf32> {onnx.name = "data"}, %arg1: memref<1x3x4x4xf32> {onnx.name = "output"}) {
  // CHECK-NEXT: [[C4:%[0-9a-zA-Z_]*]] = arith.constant 4 : index
  // CHECK-NEXT: [[C3:%[0-9a-zA-Z_]*]] = arith.constant 3 : index
  // CHECK-NEXT: [[C1:%[0-9a-zA-Z_]*]] = arith.constant 1 : index
  // CHECK-NEXT: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: [[CST0:%[0-9a-zA-Z_]*]] = arith.constant 3.40282347E+38 : f32
  // CHECK-NEXT: [[C0:%[0-9a-zA-Z_]*]] = arith.constant 0 : index
  // CHECK-NEXT: scf.for [[inx1:%[0-9a-zA-Z_]*]] = [[C0]] to [[C3]] step [[C1]] {
  // CHECK-NEXT:   scf.for [[inx2:%[0-9a-zA-Z_]*]] = [[C0]] to [[C4]] step [[C1]] {
  // CHECK-NEXT:     scf.for [[inx3:%[0-9a-zA-Z_]*]] = [[C0]] to [[C4]] step [[C1]] {
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x3x4x4xf32>) outs(%arg1 : memref<1x3x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
  // CHECK-NEXT:       [[V0:%[0-9a-zA-Z_]*]] = memref.load [[ARG0]][[[C0]], [[inx1]], [[inx2]], [[inx3]]] : memref<1x3x4x4xf32>
  // CHECK-NEXT:       [[V1:%[0-9a-zA-Z_]*]] = arith.cmpf ult, [[V0]], [[CST0]] : f32
  // CHECK-NEXT:       [[V2:%[0-9a-zA-Z_]*]] = arith.select [[V1]], [[V0]], [[CST0]] : f32
  // CHECK-NEXT:       [[V3:%[0-9a-zA-Z_]*]] = arith.cmpf ugt, [[V2]], [[CST]] : f32
  // CHECK-NEXT:       [[V4:%[0-9a-zA-Z_]*]] = arith.select [[V3]], [[V2]], [[CST]] : f32
  // CHECK-NEXT:       memref.store [[V4]], [[ARG1]][[[C0]], [[inx1]], [[inx2]], [[inx3]]] : memref<1x3x4x4xf32>
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 3.40282347E+38 : f32
      %0 = arith.minimumf %in, %cst_0 : f32
      %1 = arith.maximumf %0, %cst : f32
      linalg.yield %1 : f32
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
    }
  // CHECK-NEXT: return
    return
  }
}

