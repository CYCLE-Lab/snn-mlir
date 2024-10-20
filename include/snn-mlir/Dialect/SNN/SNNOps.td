#ifndef SNN_OPS_TD
#define SNN_OPS_TD

include "mlir/IR/OpBase.td"
include "snn-mlir/Dialect/SNN/SNNDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// mnemonic 指名字
class SNNOp<string mnemonic, list<Trait> traits = []> :
  Op<SNNDialect, mnemonic, traits>;

def lifOp : SNNOp<"lif", [Pure]> {
  let summary = "Leaky Integrate-and-Fire (LIF) neuron operation";
  let description = [{
    This operation simulates the behavior of a Leaky Integrate-and-Fire (LIF) neuron. 
  }];

  let arguments = (ins
    F32Tensor:$voltage,  // input tensor representing the neuron voltage
    F32Tensor:$inputs     // input tensor representing the input current
  );

  let results = (outs
    F32Tensor:$output    // output tensor representing the updated voltage
  );
}



def ifOp : SNNOp<"if", [Pure]> {
  let summary = "Integrate-and-Fire (IF) neuron operation";
  let description = [{
    This operation simulates the behavior of a Integrate-and-Fire (IF) neuron. 
  }];

  let arguments = (ins
    F32Tensor:$voltage,  // input tensor representing the neuron voltage
    F32Tensor:$inputs     // input tensor representing the input current
  );

  let results = (outs
    F32Tensor:$output    // output tensor representing the updated voltage
  );
}

#endif
