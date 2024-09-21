/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SNN.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace snn {

class SNNDialect : public ::mlir::Dialect {
  explicit SNNDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~SNNDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("snn");
  }
};
} // namespace snn
MLIR_DECLARE_EXPLICIT_TYPE_ID(::snn::SNNDialect)
