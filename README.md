SNN-MLIR (Super Neural Network based on Multi-Level Intermediate Representation)

一、Install SNN-MLIR via LLVM external projects mechanism

1. Install Requirements
```
sudo apt-get install git
sudo apt-get install ninja-build
sudo apt-get install python3
sudo pip3 install lit numpy pybind11

git submodule update --init --recursive
```

2. Build MLIR and SNN-MLIR:
```
mkdir build

cd build

cmake -G Ninja ../third_party/llvm-project/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_EXTERNAL_PROJECTS=snn-mlir \
    -DLLVM_EXTERNAL_SNN_MLIR_SOURCE_DIR=.. 

ninja

ninja check-snn-lit
```

二、Install SNN-MLIR With Python Binding

1. Install Requirements
```
sudo apt-get install git
sudo apt-get install ninja-build
sudo apt-get install python3
sudo pip3 install lit numpy pybind11

git submodule update --init --recursive
```

2. Build MLIR:

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
```

3. Build SNN-MLIR:

```
mkdir build

cd build

cmake -G Ninja .. \
  -DLLVM_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DSNN_MLIR_ENABLE_BINDINGS_PYTHON=ON

ninja

ninja check-snn-lit
```