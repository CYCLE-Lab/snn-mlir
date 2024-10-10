module {
  func.func @test_lif(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant dense<9.900000e-01> : tensor<2x2xf32>
    %0 = arith.mulf %arg0, %cst : tensor<2x2xf32>
    %1 = arith.addf %0, %arg1 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}

