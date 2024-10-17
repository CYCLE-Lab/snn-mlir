SNN-MLIR (Super Neural Network based on Multi-Level Intermediate Representation)

## 一、Install SNN-MLIR via LLVM external projects mechanism

### 1. Install Requirements
```
sudo apt-get install git
sudo apt-get install ninja-build
sudo apt-get install python3
sudo pip3 install lit numpy pybind11

git submodule update --init --recursive
```

### 2. Build MLIR and SNN-MLIR:
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

## 二、Install SNN-MLIR With Python Binding

### 1. Install Requirements
```
sudo apt-get install git
sudo apt-get install ninja-build
sudo apt-get install python3
sudo pip3 install lit numpy pybind11

git submodule update --init --recursive
```

### 2. Build MLIR:

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

### 3. Build SNN-MLIR:

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

## 三、Using Code format-all Plugin

### 1. Install Requirements

```
sudo apt install clang-format
pip install cmake-format
pip install black
```
### 2. Run format-all

```
cd build
cmake --build . --target format-all
```

### 3. Automatically Run format-all after each commit

[notes:This subsection is from ChatGPT.]

To automatically run format-all after each commit, you can use Git hooks. Specifically, you’ll want to create a post-commit hook that triggers the formatting command. Here’s how to set it up:

#### 1.	Navigate to Your Git Repository:
Open your terminal and navigate to the root directory of your Git repository.

```
cd .git/hooks
```

#### 2.	Create the Post-Commit Hook:
Inside your repository, navigate to the .git/hooks directory:

Create a new file named post-commit (or edit it if it already exists):

```
touch post-commit
```

#### 3.	Edit the Post-Commit Hook:
Open the post-commit file with a text editor and add the following content:

```
#!/bin/bash

# Run format-all after each commit
cmake --build .. --target format-all
```

Make sure to replace .. with the correct path to your build directory if it’s different.

#### 4.	Make the Hook Executable:
Run the following command to make the post-commit script executable:

```
chmod +x post-commit
```

#### 5.	Testing the Hook:
Now, whenever you make a commit, the post-commit hook will automatically run `cmake --build .. --target format-all`.

#### 6. Optional: Additional Considerations

•	Commit Messages: Since format-all might modify files, you may want to handle commit messages or consider running the format before the commit (using a pre-commit hook instead).
•	Pre-commit Hook (Optional): If you’d like to run the formatting before the commit and prevent the commit if there are changes, create a pre-commit hook instead, similar to the post-commit hook:

```
#!/bin/bash

# Run format-all before committing
cmake --build .. --target format-all

# Check if any files were modified by format-all
if ! git diff --exit-code; then
    echo "Code was formatted, please review changes."
    exit 1
fi
```
This way, you can ensure that your code style is maintained before each commit.