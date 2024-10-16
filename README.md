SNN-MLIR



1. Build LLVM and MLIR:

```
mkdir third_party/llvm-project/build


cd third_party/llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF

ninja

ninja check-mlir
```

2. Build SNN-MLIR:

```
mkdir build

cd build

cmake -G Ninja .. \
  -DLLVM_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DSNNMLIR_ENABLE_BINDINGS_PYTHON=ON

ninja

ninja snn-opt

ninja check-snn-lit
```