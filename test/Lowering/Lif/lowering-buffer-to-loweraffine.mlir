// RUN: snn-opt --split-input-file --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-tile="tile-size=32" --fold-memref-alias-ops --lower-affine --cse --promote-buffers-to-stack --arith-expand --sccp --canonicalize="top-down" --symbol-dce --canonicalize %s | FileCheck %s


#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // CHECK-LABEL: func.func @test_lif
  // CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
  // CHECK-SAME: [[ARG2:%[0-9a-zA-Z_]*]]:
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    // CHECK: [[C1:%[0-9a-zA-Z_]*]] = arith.constant 1 : index
    // CHECK: [[C2:%[0-9a-zA-Z_]*]] = arith.constant 2 : index
    // CHECK: [[C0:%[0-9a-zA-Z_]*]] = arith.constant 0 : index
    // CHECK: [[CST:%[0-9a-zA-Z_]*]] = arith.constant 0.000000e+00 : f32
    // CHECK: [[CST0:%[0-9a-zA-Z_]*]] = arith.constant 1.000000e+00 : f32
    // CHECK: [[CST1:%[0-9a-zA-Z_]*]] = arith.constant 9.900000e-01 : f32
    // CHECK: [[ALLOCA:%[0-9a-zA-Z_]*]] = memref.alloca() : memref<1x1xf32>
    // CHECK: [[ALLOCA2:%[0-9a-zA-Z_]*]] = memref.alloca() : memref<1x1xf32>
    // CHECK: [[ALLOCA3:%[0-9a-zA-Z_]*]] = memref.alloca() : memref<1x1xf32>
    // CHECK: [[ALLOCA4:%[0-9a-zA-Z_]*]] = memref.alloca() : memref<1x1xf32>
    %cst = arith.constant 9.900000e-01 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<2x2xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    linalg.mul ins(%arg0, %alloc : memref<2x2xf32>, memref<2x2xf32>) outs(%alloc_0 : memref<2x2xf32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    linalg.add ins(%arg1, %alloc_0 : memref<2x2xf32>, memref<2x2xf32>) outs(%alloc_1 : memref<2x2xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    %cst_3 = arith.constant 1.000000e+00 : f32
    linalg.fill ins(%cst_3 : f32) outs(%alloc_2 : memref<2x2xf32>)
    // CHECK: scf.for [[ARG3:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:   scf.for [[ARG4:%[0-9a-zA-Z_]*]] = [[C0]] to [[C2]] step [[C1]] {
    // CHECK:     memref.store [[CST1]], [[ALLOCA3]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V0:%[0-9a-zA-Z_]*]] = memref.load [[ARG0]][[[ARG3]], [[ARG4]]] : memref<2x2xf32>
    // CHECK:     [[V1:%[0-9a-zA-Z_]*]] = memref.load [[ALLOCA3]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V2:%[0-9a-zA-Z_]*]] = arith.mulf [[V0]], [[V1]] : f32
    // CHECK:     memref.store [[V2]], [[ALLOCA4]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V3:%[0-9a-zA-Z_]*]] = memref.load [[ARG1]][[[ARG3]], [[ARG4]]] : memref<2x2xf32>
    // CHECK:     [[V4:%[0-9a-zA-Z_]*]] = memref.load [[ALLOCA4]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V5:%[0-9a-zA-Z_]*]] = arith.addf [[V3]], [[V4]] : f32
    // CHECK:     memref.store [[V5]], [[ALLOCA]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     memref.store [[CST0]], [[ALLOCA2]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V6:%[0-9a-zA-Z_]*]] = memref.load [[ALLOCA2]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V7:%[0-9a-zA-Z_]*]] = memref.load [[ALLOCA]][[[C0]], [[C0]]] : memref<1x1xf32>
    // CHECK:     [[V8:%[0-9a-zA-Z_]*]] = arith.cmpf olt, [[V7]], [[V6]] : f32
    // CHECK:     [[V9:%[0-9a-zA-Z_]*]] = arith.select [[V8]], [[CST]], [[CST0]] : f32
    // CHECK:     memref.store [[V9]], [[ARG2]][[[ARG3]], [[ARG4]]] : memref<2x2xf32>
    // CHECK:   }
    // CHECK: } 
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_2, %alloc_1 : memref<2x2xf32>, memref<2x2xf32>) outs(%arg2 : memref<2x2xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %0 = arith.cmpf olt, %in_4, %in : f32
      %cst_5 = arith.constant 0.000000e+00 : f32
      %1 = arith.select %0, %cst_5, %cst_3 : f32
      linalg.yield %1 : f32
    }
    // CHECK: return
    return
  }
}

