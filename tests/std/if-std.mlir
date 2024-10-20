#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_if(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.add ins(%arg1, %arg0 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = tensor.empty() : tensor<2x2xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.cmpf olt, %in_0, %in : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %7 = arith.select %6, %cst_1, %cst_2 : f32
      linalg.yield %7 : f32
    } -> tensor<2x2xf32>
    return %5 : tensor<2x2xf32>
  }
}

