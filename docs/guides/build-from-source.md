# Build from Source

Compile llama.cpp binaries for llcuda from source.

## Prerequisites

- CUDA 12.x toolkit
- CMake 3.18+
- GCC 11+

## Clone llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout b7760
```

## Build with CUDA

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_FLASH_ATTN=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

## Package Binaries

```bash
mkdir -p llcuda-binaries/bin llcuda-binaries/lib
cp build/bin/llama-* llcuda-binaries/bin/
cp build/*.so llcuda-binaries/lib/
tar -czf llcuda-v2.2.0-custom.tar.gz llcuda-binaries/
```

## Use Custom Binaries

```python
import os
os.environ["LLCUDA_BINARY_PATH"] = "/path/to/llcuda-binaries"

from llcuda.server import ServerManager
# Will use custom binaries
```

## See Also

- [Installation Guide](installation.md)
