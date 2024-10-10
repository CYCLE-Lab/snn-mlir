#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_lif(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 9.900000e-01 : f32
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = tensor.empty() : tensor<2x2xf32>
    %3 = linalg.mul ins(%arg0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.add ins(%arg1, %3 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = tensor.empty() : tensor<2x2xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %8 = tensor.empty() : tensor<2x2xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %4 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%8 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %10 = arith.cmpf olt, %in_1, %in : f32
      %cst_2 = arith.constant 0.000000e+00 : f32
      %cst_3 = arith.constant 1.000000e+00 : f32
      %11 = arith.select %10, %cst_2, %cst_3 : f32
      linalg.yield %11 : f32
    } -> tensor<2x2xf32>
    return %9 : tensor<2x2xf32>
  }
}

