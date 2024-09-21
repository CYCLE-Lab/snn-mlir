module {
  func.func @test_lif(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 0.00999999977 : f32
    %splat = tensor.splat %cst : tensor<2x2xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = arith.subf %cst_0, %cst : f32
    %splat_1 = tensor.splat %0 : tensor<2x2xf32>
    %1 = arith.mulf %arg0, %splat_1 : tensor<2x2xf32>
    %cst_2 = arith.constant 1.000000e+00 : f32
    %splat_3 = tensor.splat %cst_2 : tensor<2x2xf32>
    %2 = arith.mulf %arg1, %splat_3 : tensor<2x2xf32>
    %3 = arith.addf %1, %2 : tensor<2x2xf32>
    %cst_4 = arith.constant 1.000000e+00 : f32
    %splat_5 = tensor.splat %cst_4 : tensor<2x2xf32>
    %4 = arith.cmpf ogt, %3, %splat_5 : tensor<2x2xf32>
    %true = arith.constant true
    scf.if %true {
      %cst_6 = arith.constant 0.000000e+00 : f32
    }
    return %3 : tensor<2x2xf32>
  }
}

