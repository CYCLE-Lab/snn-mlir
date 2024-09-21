func.func @test_lif(%voltage: tensor<2x2xf32>, %input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %output = "snn.lif"(%voltage, %input) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %output2 = "snn.lif"(%voltage, %output) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %output : tensor<2x2xf32>
}
