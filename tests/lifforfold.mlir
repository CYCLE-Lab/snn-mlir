module {
  func.func @test_lif(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense<9.900000e-01> : tensor<2x2xf32>
    %0 = arith.mulf %arg0, %cst_2 : tensor<2x2xf32>
    %1 = arith.addf %0, %arg1 : tensor<2x2xf32>
    %2 = arith.cmpf ogt, %1, %cst : tensor<2x2xf32>
    %3 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %1) -> (tensor<2x2xf32>) {
      %4 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x2xf32>) {
        %extracted = tensor.extract %2[%arg2, %arg4] : tensor<2x2xi1>
        %5 = scf.if %extracted -> (tensor<2x2xf32>) {
          %inserted = tensor.insert %cst_1 into %arg5[%arg2, %arg4] : tensor<2x2xf32>
          scf.yield %inserted : tensor<2x2xf32>
        } else {
          %inserted = tensor.insert %cst_0 into %arg5[%arg2, %arg4] : tensor<2x2xf32>
          scf.yield %inserted : tensor<2x2xf32>
        }
        scf.yield %5 : tensor<2x2xf32>
      }
      scf.yield %4 : tensor<2x2xf32>
    }
    return %3 : tensor<2x2xf32>
  }
}

