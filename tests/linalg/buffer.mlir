#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_lif(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
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
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_2, %alloc_1 : memref<2x2xf32>, memref<2x2xf32>) outs(%arg2 : memref<2x2xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %0 = arith.cmpf olt, %in_4, %in : f32
      %cst_5 = arith.constant 0.000000e+00 : f32
      %cst_6 = arith.constant 1.000000e+00 : f32
      %1 = arith.select %0, %cst_5, %cst_6 : f32
      linalg.yield %1 : f32
    }
    return
  }
}

