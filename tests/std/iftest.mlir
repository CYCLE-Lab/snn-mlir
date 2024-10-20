func.func @test_if(%voltage: tensor<2x2xf32>, %input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %output = "snn.if"(%voltage, %input) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %output : tensor<2x2xf32>
}
