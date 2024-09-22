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
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c2_6 = arith.constant 2 : index
    %cst_7 = arith.constant 1.000000e+00 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c2_6 step %c1 {
        %extracted = tensor.extract %4[%arg2, %arg3] : tensor<2x2xi1>
        %5 = scf.if %extracted -> (tensor<2x2xf32>) {
          %inserted = tensor.insert %cst_7 into %3[%arg2, %arg3] : tensor<2x2xf32>
          scf.yield %inserted : tensor<2x2xf32>
        } else {
          %inserted = tensor.insert %cst_8 into %3[%arg2, %arg3] : tensor<2x2xf32>
          scf.yield %inserted : tensor<2x2xf32>
        }
      }
    }
    return %3 : tensor<2x2xf32>
  }
}

