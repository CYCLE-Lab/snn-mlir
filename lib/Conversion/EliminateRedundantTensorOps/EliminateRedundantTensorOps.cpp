//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "snn-mlir/Conversion/EliminateRedundantTensorOps/EliminateRedundantTensorOpspasses.h"
#include "snn-mlir/Dialect/SNN/SNNDialect.h"


using namespace mlir;
using namespace snn;

namespace {
class EliminateRedundantTensorOpsPass
    : public PassWrapper<EliminateRedundantTensorOpsPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto function = getOperation();
    mlir::PatternRewriter rewriter(function.getContext());

    // Save the operation to be deleted
    SmallVector<Operation *, 8> opsToDelete;

    // Remove redundant expand_shape and extract_slice operations
    function.walk([&](Operation *op) {
      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
        // Determine if elimination is possible
        if (shouldEliminate_e(expandOp)) {
          // Traverse all users of the current expandOp, update references
          for (Operation *user : expandOp.getResult().getUsers()) {
            if (isa<tensor::ExtractSliceOp>(user)) {
              for (Operation *userOfUser : user->getUsers()) {
                for (unsigned i = 0; i < userOfUser->getNumOperands(); ++i) {
                  if (userOfUser->getOperand(i) == user->getResult(0)) {
                    userOfUser->setOperand(i, expandOp.getSrc());
                  }
                }
              }
              opsToDelete.push_back(user);
            } else {
              // If it is another Op, directly update its input
              for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                if (user->getOperand(i) == expandOp.getResult()) {
                  user->setOperand(i, expandOp.getSrc());
                }
              }
            }
          }
          opsToDelete.push_back(expandOp);
        }
      }

      // Check the insert_slice operation
      else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
        if (shouldEliminate_i(insertOp)) {
          for (Operation *user : insertOp.getResult().getUsers()) {
            if (isa<tensor::CollapseShapeOp>(user)) {
              for (Operation *userOfUser : user->getUsers()) {
                for (unsigned i = 0; i < userOfUser->getNumOperands(); ++i) {
                  if (userOfUser->getOperand(i) == user->getResult(0)) {
                    userOfUser->setOperand(i, insertOp.getSource());
                  }
                }
              }
              opsToDelete.push_back(user);
            } else {
              for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                if (user->getOperand(i) == insertOp.getResult()) {
                  user->setOperand(i, insertOp.getSource());
                }
              }
            }
          }
          opsToDelete.push_back(insertOp);
        }
      }
    });

    for (Operation *op : opsToDelete) {
      rewriter.eraseOp(op);
    }
  }

  bool shouldEliminate_e(tensor::ExpandShapeOp op) {
    // Check the users of the operation
    for (Operation *user : op.getResult().getUsers()) {
      // Check for the existence of extract_slice
      if (isa<tensor::ExtractSliceOp>(user)) {
        auto sourceType = op.getSrc().getType();
        auto userType = user->getResult(0).getType();

        // Determine if the extract type can be eliminated
        if (userType == sourceType) {
          return true;
        }
      }
    }
    return false;
  }

  bool shouldEliminate_i(tensor::InsertSliceOp op) {
    for (Operation *user : op.getResult().getUsers()) {
      if (isa<tensor::CollapseShapeOp>(user)) {
        auto sourceType = op.getSource().getType();
        auto userType = user->getResult(0).getType();

        if (userType == sourceType) {
          return true;
        }
      }
    }
    return false;
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
  }

  StringRef getArgument() const final {
    return "eliminate-redundant-tensorops";
  }

  StringRef getDescription() const final {
    return "Eliminate redundant tensor.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> snn::createEliminateRedundantTensorOpsPass() {
  return std::make_unique<EliminateRedundantTensorOpsPass>();
}

static mlir::PassRegistration<EliminateRedundantTensorOpsPass> pass;