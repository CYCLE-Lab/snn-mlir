#include "SNN/SNNDialect.h"
#include "SNN/SNNOps.h"

#include "SNN/SNNDialect.cpp.inc"
#define GET_OP_CLASSES
#include "SNN/SNN.cpp.inc"

using namespace mlir;
using namespace snn;

void SNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SNN/SNN.cpp.inc"
  >();
}
