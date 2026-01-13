# Building CUDA Binaries for llcuda

This tutorial shows how to build optimized CUDA binaries compatible with llcuda v2.1.0+ on Tesla T4 GPUs. The binaries are built from llama.cpp with FlashAttention and CUDA 12 support.

!!! warning "Advanced Topic"
    This guide is for advanced users who want to customize binaries or contribute to llcuda development. Regular users should use the pre-built binaries that auto-download from GitHub Releases.

## Overview

llcuda v2.1.0 uses pre-built v2.0.6 CUDA binaries based on llama.cpp with these optimizations:

!!! info "Binary Compatibility"
    The v2.0.6 binaries are fully compatible with llcuda v2.1.0 and later versions. The binary format remains stable while the Python API receives new features.

- **FlashAttention** - 2-3x faster attention for long contexts
- **CUDA Graphs** - Reduced kernel launch overhead
- **Tensor Cores** - INT4/INT8 hardware acceleration (SM 7.5)
- **cuBLAS** - Optimized matrix multiplication
- **SM 7.5 targeting** - Tesla T4 specific optimizations

## Prerequisites

### System Requirements

- **GPU:** Tesla T4 (SM 7.5) - Google Colab recommended
- **CUDA:** 12.0 or higher (12.4 recommended)
- **GCC:** 11.x or 12.x
- **CMake:** 3.18+
- **Disk Space:** 5 GB for build artifacts
- **RAM:** 8 GB minimum
- **Time:** 20-30 minutes on T4

### Software Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget

# Verify CUDA installation
nvcc --version
nvidia-smi

# Verify GCC version
gcc --version  # Should be 11.x or 12.x
```

## Build Process

### Method 1: Using Google Colab Notebook

The easiest way to build binaries is using the provided Colab notebook:

1. **Open the build notebook:**
   - [build_llcuda_v2_t4_colab.ipynb](https://github.com/waqasm86/llcuda/blob/main/notebooks/build_llcuda_v2_t4_colab.ipynb)

2. **Select T4 GPU runtime:**
   - Runtime > Change runtime type > GPU > T4

3. **Run all cells:**
   - Runtime > Run all

4. **Download built binaries:**
   - Files will be in `/content/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz`

### Method 2: Manual Build

For local systems with Tesla T4:

#### Step 1: Clone llama.cpp

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Checkout stable commit (optional but recommended)
git checkout b1698  # Replace with known-good commit
```

#### Step 2: Configure CMake for T4

```bash
# Create build directory
mkdir build
cd build

# Configure with CUDA 12 and FlashAttention
cmake .. \
  -DLLAMA_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_DMMV_F16=ON \
  -DGGML_CUDA_FORCE_MMQ=OFF \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Key CMake flags explained:**

| Flag | Purpose |
|------|---------|
| `LLAMA_CUDA=ON` | Enable CUDA support |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | FlashAttention for all quantization types |
| `GGML_CUDA_GRAPHS=ON` | CUDA graphs for reduced overhead |
| `CMAKE_CUDA_ARCHITECTURES=75` | Target Tesla T4 (SM 7.5) |
| `GGML_CUDA_FORCE_CUBLAS=ON` | Use cuBLAS for matrix operations |

#### Step 3: Build Binaries

```bash
# Build with parallel jobs
cmake --build . --config Release -j$(nproc)

# Expected output:
# [100%] Built target llama-server
# [100%] Built target llama-cli
# [100%] Built target llama-quantize
# [100%] Built target llama-embedding
```

**Build time:** ~20-25 minutes on T4

#### Step 4: Verify Built Binaries

```bash
# Check binaries exist
ls -lh bin/

# Expected files:
# llama-server       (~6.5 MB)
# llama-cli          (~4.2 MB)
# llama-quantize     (~434 KB)
# llama-embedding    (~3.3 MB)
# llama-bench        (~581 KB)

# Check library files
ls -lh src/

# Expected libraries:
# libggml-cuda.so    (~221 MB)
# libllama.so        (~15 MB)
# libggml-base.so    (~8 MB)
# libggml-cpu.so     (~6 MB)
```

#### Step 5: Test Binary

```bash
# Quick test (should show CUDA support)
./bin/llama-server --version

# Expected output:
# llama-server: built with CUDA 12.4 for compute capability 7.5
# FlashAttention: enabled
# CUDA graphs: enabled
```

### Method 3: Using Build Script

llcuda includes a build script for automation:

```bash
# Clone llcuda repository
git clone https://github.com/waqasm86/llcuda.git
cd llcuda/scripts

# Run build script
chmod +x build_t4_binaries.sh
./build_t4_binaries.sh

# Binaries will be in ../build-artifacts/
```

## Creating Distribution Package

After building, create a distribution package:

### Step 1: Organize Files

```bash
# Create package directory structure
mkdir -p llcuda-binaries-cuda12-t4/bin
mkdir -p llcuda-binaries-cuda12-t4/lib

# Copy binaries
cd llama.cpp/build
cp bin/llama-server ../../../llcuda-binaries-cuda12-t4/bin/
cp bin/llama-cli ../../../llcuda-binaries-cuda12-t4/bin/
cp bin/llama-quantize ../../../llcuda-binaries-cuda12-t4/bin/
cp bin/llama-embedding ../../../llcuda-binaries-cuda12-t4/bin/
cp bin/llama-bench ../../../llcuda-binaries-cuda12-t4/bin/

# Copy libraries
cp src/libggml-cuda.so ../../../llcuda-binaries-cuda12-t4/lib/
cp src/libllama.so ../../../llcuda-binaries-cuda12-t4/lib/
cp src/libggml-base.so ../../../llcuda-binaries-cuda12-t4/lib/
cp src/libggml-cpu.so ../../../llcuda-binaries-cuda12-t4/lib/
```

### Step 2: Create Tarball

```bash
# Create compressed archive
cd ../../..
tar -czf llcuda-binaries-cuda12-t4-v2.0.6.tar.gz llcuda-binaries-cuda12-t4/

# Verify size (~266 MB)
ls -lh llcuda-binaries-cuda12-t4-v2.0.6.tar.gz

# Calculate SHA256 checksum
sha256sum llcuda-binaries-cuda12-t4-v2.0.6.tar.gz > llcuda-binaries-cuda12-t4-v2.0.6.tar.gz.sha256
```

### Step 3: Test Package

```bash
# Extract to test
mkdir test-extract
cd test-extract
tar -xzf ../llcuda-binaries-cuda12-t4-v2.0.6.tar.gz

# Test llama-server
export LD_LIBRARY_PATH=llcuda-binaries-cuda12-t4/lib:$LD_LIBRARY_PATH
./llcuda-binaries-cuda12-t4/bin/llama-server --version

# Should show CUDA 12.x, SM 7.5, FlashAttention enabled
```

## Build Configuration Options

### FlashAttention Variants

```bash
# FlashAttention for all quantizations (default)
-DGGML_CUDA_FA_ALL_QUANTS=ON

# FlashAttention for FP16 only (smaller binary)
-DGGML_CUDA_FA_ALL_QUANTS=OFF

# Disable FlashAttention (not recommended)
# Remove -DGGML_CUDA_FA_ALL_QUANTS flag
```

### Compute Capability Targeting

```bash
# Tesla T4 only (SM 7.5)
-DCMAKE_CUDA_ARCHITECTURES=75

# Multiple architectures (larger binary)
-DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

# All modern architectures
-DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
```

### Memory Optimizations

```bash
# Use FP16 for DMMV (faster, more memory)
-DGGML_CUDA_DMMV_F16=ON

# Force matrix multiplication kernels
-DGGML_CUDA_FORCE_MMQ=OFF

# Use cuBLAS for better performance
-DGGML_CUDA_FORCE_CUBLAS=ON
```

## Verifying Build Quality

### Check CUDA Features

```bash
# Run llama-server with --help
./bin/llama-server --help | grep -i cuda

# Should show:
# --flash-attn               enable FlashAttention
# --cuda-graphs              use CUDA graphs for better performance
```

### Performance Test

```bash
# Download a small test model
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Run benchmark
./bin/llama-bench \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 99 \
  -p 512 \
  -n 128

# Expected output should show:
# - GPU layers: 99
# - Tokens/sec > 200
# - Using CUDA compute 7.5
```

### Memory Bandwidth Test

```bash
# Check if using Tensor Cores
./bin/llama-server \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 99 \
  -v

# Look for log output:
# "using Tensor Cores"
# "FlashAttention enabled"
```

## Troubleshooting Build Issues

### CMake Can't Find CUDA

```bash
# Set CUDA path explicitly
export CUDA_PATH=/usr/local/cuda-12.4
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Retry CMake
cmake .. -DLLAMA_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc
```

### GCC Version Mismatch

```bash
# CUDA 12.x requires GCC 11 or 12
sudo apt-get install gcc-11 g++-11

# Use specific GCC version
cmake .. -DLLAMA_CUDA=ON -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
```

### Out of Memory During Build

```bash
# Reduce parallel jobs
cmake --build . --config Release -j2

# Or build sequentially
cmake --build . --config Release -j1
```

### Missing libcudart.so

```bash
# Add CUDA lib to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
ldd bin/llama-server | grep cuda
```

## Contributing Built Binaries

If you build optimized binaries for different configurations:

1. **Test thoroughly:**
   - Run on clean Colab instance
   - Verify all models load correctly
   - Benchmark performance

2. **Create GitHub issue:**
   - Describe build configuration
   - Share build flags used
   - Include benchmark results

3. **Upload to GitHub Release:**
   - Fork llcuda repository
   - Create pull request with build documentation

## Next Steps

- [Performance Optimization](performance.md) - Tune runtime parameters
- [Benchmarks](../performance/benchmarks.md) - Compare performance
- [Troubleshooting](../guides/troubleshooting.md) - Fix common issues
- [Unsloth Integration](unsloth-integration.md) - Deploy fine-tuned models

## Resources

- **Build Notebook:** [build_llcuda_v2_t4_colab.ipynb](https://github.com/waqasm86/llcuda/blob/main/notebooks/build_llcuda_v2_t4_colab.ipynb)
- **llama.cpp:** [GitHub](https://github.com/ggerganov/llama.cpp)
- **CUDA Toolkit:** [Download](https://developer.nvidia.com/cuda-downloads)
- **CMake Documentation:** [cmake.org](https://cmake.org/documentation/)

## Reference Build Configuration

For reproducible builds, here's the exact configuration used for llcuda v2.0.6 binaries (compatible with v2.1.0+):

```bash
cmake .. \
  -DLLAMA_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_DMMV_F16=ON \
  -DGGML_CUDA_FORCE_MMQ=OFF \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc

cmake --build . --config Release -j$(nproc)
```

**Environment:**
- CUDA: 12.4
- GCC: 11.4
- CMake: 3.27
- llama.cpp: commit b1698
- GPU: Tesla T4
