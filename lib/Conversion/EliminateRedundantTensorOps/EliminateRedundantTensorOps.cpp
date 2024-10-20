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

    // 保存待删除的操作
    SmallVector<Operation *, 8> opsToDelete;

    // 删除冗余的 expand_shape 与 extract_slice
    function.walk([&](Operation *op) {
      // 检查 expand_shape 操作
      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
        // 判断是否可以消除的逻辑
        if (shouldEliminate_e(expandOp)) {
          // 变量当前expandOp的所有user,更新引用,确保后续操作正确指向原始张量
          for (Operation *user : expandOp.getResult().getUsers()) {
            // 如果是 ExtractSliceOp，则更新其user的输入
            if (isa<tensor::ExtractSliceOp>(user)) {
              for (Operation *userOfUser : user->getUsers()) {
                for (unsigned i = 0; i < userOfUser->getNumOperands(); ++i) {
                  // 将操作数替换为 expandOp 的源
                  if (userOfUser->getOperand(i) == user->getResult(0)) {
                    userOfUser->setOperand(i, expandOp.getSrc());
                  }
                }
              }
              // rewriter.eraseOp(user); // 删除不再需要的操作
              opsToDelete.push_back(user);  // 收集待删除的操作
            } else {
              // 如果是其他Op，则直接更新其输入
              for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                if (user->getOperand(i) == expandOp.getResult()) {
                  user->setOperand(i, expandOp.getSrc());
                }
              }
            }
          }
          // rewriter.eraseOp(expandOp); // 删除 expand 操作
          opsToDelete.push_back(expandOp);  // 收集待删除的操作
        }
      }

      // 检查 insert_slice 操作
      else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
        // 判断是否可以消除的逻辑
        if (shouldEliminate_i(insertOp)) {
          // 变量当前insertOp的所有user,更新引用,确保后续操作正确指向原始张量
          for (Operation *user : insertOp.getResult().getUsers()) {
            // 如果是 collapse_shape，则更新collapse_shape user的输入
            if (isa<tensor::CollapseShapeOp>(user)) {
              for (Operation *userOfUser : user->getUsers()) {
                for (unsigned i = 0; i < userOfUser->getNumOperands(); ++i) {
                  // 将操作数替换为 insertOp 的源
                  if (userOfUser->getOperand(i) == user->getResult(0)) {
                    userOfUser->setOperand(i, insertOp.getSource());
                  }
                }
              }
              // rewriter.eraseOp(user); // 删除不再需要的操作
              opsToDelete.push_back(user);  // 收集待删除的操作
            } else {
              // 如果是其他Op，则直接更新其输入
              for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                if (user->getOperand(i) == insertOp.getResult()) {
                  user->setOperand(i, insertOp.getSource());
                }
              }
            }
          }
          // rewriter.eraseOp(insertOp); // 删除 insert 操作
          opsToDelete.push_back(insertOp);  // 收集待删除的操作
        }
      }
    });

    // 删除所有待删除的操作
    for (Operation *op : opsToDelete) {
      rewriter.eraseOp(op);
    }
  }

  bool shouldEliminate_e(tensor::ExpandShapeOp op) {
    // 检查操作的用户
    for (Operation *user : op.getResult().getUsers()) {
      // 检查是否有 extract_slice
      if (isa<tensor::ExtractSliceOp>(user)) {
        auto sourceType = op.getSrc().getType();
        auto userType = user->getResult(0).getType();

        // 判断 extract 类型是否可以被消除
        if (userType == sourceType) {
          return true;  // 可以消除
        }
      }
    }
    return false;  // 不可消除
  }

  bool shouldEliminate_i(tensor::InsertSliceOp op) {
    // 检查操作的用户
    for (Operation *user : op.getResult().getUsers()) {
      // 检查是否有 collapse_shape
      if (isa<tensor::CollapseShapeOp>(user)) {
        auto sourceType = op.getSource().getType();
        auto userType = user->getResult(0).getType();

        // 判断 collapse_shape 类型是否可以被消除
        if (userType == sourceType) {
          return true;  // 可以消除
        }
      }
    }
    return false;  // 不可消除
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