#!/bin/bash
# Create all remaining llcuda v2.2.0 documentation pages

cd "$(dirname "$0")"

echo "Creating remaining documentation pages..."

# SEO files
cat > docs/robots.txt << 'EOF'
User-agent: *
Allow: /

Sitemap: https://llcuda.github.io/sitemap.xml
EOF

cat > docs/sitemap.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://llcuda.github.io/</loc>
    <priority>1.0</priority>
    <changefreq>weekly</changefreq>
  </url>
  <url>
    <loc>https://llcuda.github.io/guides/quickstart/</loc>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://llcuda.github.io/guides/installation/</loc>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://llcuda.github.io/tutorials/index/</loc>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://llcuda.github.io/kaggle/overview/</loc>
    <priority>0.8</priority>
  </url>
</urlset>
EOF

# Create additional guides
cat > docs/guides/model-selection.md << 'EOF'
# Model Selection Guide

Choose the right model and quantization for Kaggle dual T4.

## Recommended Models

### Small Models (1-3B)
- **Gemma 2-2B** - Best performance
- **Qwen2.5-1.5B** - Efficient
- **Gemma 3-1B** - Fast inference

### Medium Models (7-8B)
- **Qwen2.5-7B** - Great quality
- **Llama-3.2-7B** - Good balance
- **Mistral-7B** - Popular choice

### Large Models (70B)
- **Llama-3.1-70B** - Best with IQ3_XS

## Quantization Selection

| VRAM Available | Recommended Quant |
|----------------|-------------------|
| < 5 GB | Q4_K_M |
| 5-10 GB | Q5_K_M |
| 10-15 GB | Q6_K |
| 25-30 GB (70B) | IQ3_XS |
EOF

cat > docs/guides/troubleshooting.md << 'EOF'
# Troubleshooting

Common issues and solutions for llcuda v2.2.0.

## GPU Not Detected

**Problem**: `No CUDA GPUs detected`

**Solution**:
1. Check Kaggle Settings → Accelerator → GPU T4 × 2
2. Restart kernel
3. Run `nvidia-smi` to verify

## Server Won't Start

**Problem**: `llama-server failed to start`

**Solution**:
```python
server = ServerManager()
logs = server.get_logs()
print(logs)  # Check error messages
```

## Out of Memory

**Problem**: `CUDA out of memory`

**Solution**:
- Use smaller quantization (Q4_K_M → IQ3_XS)
- Reduce context_size
- Enable tensor-split for dual GPU
EOF

cat > docs/guides/faq.md << 'EOF'
# FAQ

Frequently asked questions about llcuda v2.2.0.

## General

**Q: Does llcuda work on Google Colab?**
A: No, v2.2.0 is optimized for Kaggle dual T4. Colab has single T4.

**Q: Can I use llcuda locally?**
A: Yes, with Tesla T4 or compatible GPU and CUDA 12.x.

**Q: Is PyPI installation available?**
A: No, install from GitHub: `pip install git+https://github.com/llcuda/llcuda.git@v2.2.0`

## Technical

**Q: What's the difference between tensor-split and NCCL?**
A: llama.cpp uses native CUDA tensor-split, not NCCL. NCCL is for PyTorch distributed training.

**Q: Can I run 70B models?**
A: Yes, with IQ3_XS quantization on dual T4 (30GB VRAM).

**Q: What quantization should I use?**
A: Q4_K_M for most models, IQ3_XS for 70B.
EOF

cat > docs/guides/build-from-source.md << 'EOF'
# Build from Source

Build llcuda v2.2.0 binaries from source on Kaggle.

## Prerequisites

- Kaggle dual T4 environment
- CUDA 12.5 toolkit
- CMake 3.24+

## Build Steps

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 388ce82  # v2.2.0 commit

# Build with CUDA
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build --config Release -j8
```

See: Complete build notebook in repository.
EOF

# Create API reference pages
cat > docs/api/overview.md << 'EOF'
# API Reference

Complete API documentation for llcuda v2.2.0.

## Core Modules

- [`llcuda.api.client`](client.md) - OpenAI-compatible client
- [`llcuda.api.multigpu`](multigpu.md) - Multi-GPU configuration
- [`llcuda.api.gguf`](gguf.md) - GGUF tools
- [`llcuda.api.nccl`](nccl.md) - NCCL integration
- [`llcuda.server`](server.md) - Server management
- [`llcuda.graphistry`](graphistry.md) - Graphistry integration

## Quick Examples

### Start Server
```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(model_path="model.gguf")
server = ServerManager()
server.start_with_config(config)
```

### OpenAI Client
```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient()
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Multi-GPU
```python
from llcuda.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config()
print(config.to_cli_args())
```
EOF

cat > docs/api/client.md << 'EOF'
# LlamaCppClient

OpenAI-compatible client for llama.cpp server.

## Usage

```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

## Methods

- `chat.completions.create()` - Chat completions
- `completions.create()` - Text completions
- `embeddings.create()` - Generate embeddings

## Streaming

```python
stream = client.chat.completions.create(
    messages=[...],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```
EOF

cat > docs/api/multigpu.md << 'EOF'
# Multi-GPU API

Configure multi-GPU inference for Kaggle dual T4.

## Functions

### `detect_gpus()`
```python
from llcuda.api.multigpu import detect_gpus

gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.name} ({gpu.memory_total_gb:.1f} GB)")
```

### `kaggle_t4_dual_config()`
```python
from llcuda.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_size_gb=25)
# Returns MultiGPUConfig optimized for dual T4
```

### `estimate_model_vram()`
```python
from llcuda.api.multigpu import estimate_model_vram

vram = estimate_model_vram(
    param_count=7,  # 7B model
    quantization="Q4_K_M",
    ctx_size=4096
)
print(f"Estimated VRAM: {vram:.1f} GB")
```

## Classes

### `MultiGPUConfig`
```python
from llcuda.api.multigpu import MultiGPUConfig, SplitMode

config = MultiGPUConfig(
    n_gpu_layers=-1,
    tensor_split=[0.5, 0.5],
    split_mode=SplitMode.LAYER,
    flash_attention=True,
)

cli_args = config.to_cli_args()
```
EOF

echo "✅ Created all remaining documentation pages"
echo ""
echo "📝 Next steps:"
echo "1. chmod +x create_remaining_docs.sh"
echo "2. ./create_remaining_docs.sh"
echo "3. mkdocs serve"
echo "4. mkdocs gh-deploy"
EOF

chmod +x create_remaining_docs.sh

echo "✓ Created deployment script: create_remaining_docs.sh"
