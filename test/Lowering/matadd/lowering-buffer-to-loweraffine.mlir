// RUN: snn-opt --split-input-file  --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-tile="tile-size=32" --fold-memref-alias-ops --lower-affine  --cse --promote-buffers-to-stack --arith-expand --sccp --canonicalize="top-down" --symbol-dce  --canonicalize %s | FileCheck %s


#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // CHECK-LABEL: func.func @test_matadd
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_matadd(%arg0: memref<4x5xf32> {onnx.name = "x"}, %arg1: memref<4x5xf32> {onnx.name = "y"}, %arg2: memref<4x5xf32> {onnx.name = "output"}) {
    // CHECK-NEXT: [[C1:%[0-9a-zA-Z_]*]] = arith.constant 1 : index
    // CHECK-NEXT: [[C5:%[0-9a-zA-Z_]*]] = arith.constant 5 : index
    // CHECK-NEXT: [[C4:%[0-9a-zA-Z_]*]] = arith.constant 4 : index
    // CHECK-NEXT: [[C0:%[0-9a-zA-Z_]*]] = arith.constant 0 : index
    // CHECK-NEXT: scf.for [[ARG3:%[0-9a-zA-Z_]*]] = [[C0]] to [[C4]] step [[C1]] {
    // CHECK-NEXT:   scf.for [[ARG4:%[0-9a-zA-Z_]*]] = [[C0]] to [[C5]] step [[C1]] {
    // CHECK-NEXT:     [[V0:%[0-9a-zA-Z_]*]] = memref.load [[ARG0]][[[ARG3]], [[ARG4]]] : memref<4x5xf32>
    // CHECK-NEXT:     [[V1:%[0-9a-zA-Z_]*]] = memref.load [[ARG1]][[[ARG3]], [[ARG4]]] : memref<4x5xf32>
    // CHECK-NEXT:     [[V2:%[0-9a-zA-Z_]*]] = arith.addf [[V0]], [[V1]] : f32
    // CHECK-NEXT:     memref.store [[V2]], [[ARG2]][[[ARG3]], [[ARG4]]] : memref<4x5xf32>
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x5xf32>, memref<4x5xf32>) outs(%arg2 : memref<4x5xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    // CHECK-NEXT: return
    return
  }
}
