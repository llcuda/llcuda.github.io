# llcuda v2.2.0 Documentation

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/llcuda/llcuda/releases/tag/v2.2.0)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official documentation website for **llcuda v2.2.0** - CUDA 12 inference backend for Unsloth with Graphistry network visualization on Kaggle dual Tesla T4 GPUs.

🌐 **Live Documentation**: [https://llcuda.github.io/](https://llcuda.github.io/)

## What is llcuda v2.2.0?

llcuda is a CUDA 12 inference backend specifically designed for deploying [Unsloth](https://unsloth.ai/)-fine-tuned models on Kaggle's dual Tesla T4 GPUs (30GB total VRAM).

### Key Features

- 🚀 **Dual T4 Support**: Run on Kaggle's 2× Tesla T4 GPUs (15GB each)
- 🔥 **Split-GPU Architecture**: LLM on GPU 0, Graphistry on GPU 1
- ⚡ **Native CUDA tensor-split**: llama.cpp layer distribution (NOT NCCL)
- 🎯 **70B Model Support**: Run Llama-70B IQ3_XS on 30GB VRAM
- 📦 **29 GGUF Quantization Formats**: K-quants and I-quants
- 🔧 **OpenAI-compatible API**: Drop-in replacement via llama-server
- 🌐 **Graphistry Integration**: Extract and visualize knowledge graphs

### Performance Benchmarks

| Model | Quantization | VRAM | Speed | Platform |
|-------|--------------|------|-------|----------|
| Gemma 2-2B | Q4_K_M | ~3 GB | ~60 tok/s | Single T4 |
| Llama-3.2-3B | Q4_K_M | ~4 GB | ~45 tok/s | Single T4 |
| Qwen-2.5-7B | Q4_K_M | ~7 GB | ~25 tok/s | Single T4 |
| Llama-70B | IQ3_XS | ~28 GB | ~12 tok/s | Dual T4 |

## Quick Links

- 📚 [Documentation](https://llcuda.github.io/)
- 🚀 [Quick Start Guide](https://llcuda.github.io/guides/quickstart/)
- 📖 [10 Kaggle Tutorial Notebooks](https://llcuda.github.io/tutorials/index/)
- 🔧 [API Reference](https://llcuda.github.io/api/overview/)
- 💻 [Main Repository](https://github.com/llcuda/llcuda)
- 📦 [PyPI Package](https://pypi.org/project/llcuda/)

## Documentation Structure

- **Getting Started**: Installation, quick start, Kaggle setup
- **Kaggle Dual T4**: Multi-GPU inference, tensor-split, large models
- **Tutorial Notebooks**: 10 comprehensive Kaggle notebooks
- **Architecture**: Split-GPU design, LLM + Graphistry
- **Unsloth Integration**: Fine-tuning → GGUF → Deployment
- **Graphistry & Visualization**: Knowledge graph extraction
- **Performance**: Benchmarks, optimization, memory management
- **GGUF & Quantization**: K-quants, I-quants, selection guide
- **API Reference**: ServerManager, MultiGPU, GGUF tools

## Development

### Setup

```bash
# Install dependencies
pip install mkdocs-material mkdocs-minify-plugin

# Serve locally
mkdocs serve

# View at http://127.0.0.1:8000
```

### Deployment

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```

## SEO & Keywords

llcuda, CUDA 12, Tesla T4, Kaggle, dual GPU, LLM inference, Unsloth, GGUF, quantization, llama.cpp, multi-GPU, tensor-split, Graphistry, knowledge graphs, FlashAttention, 70B models, split-GPU architecture, Kaggle notebooks, RAPIDS, cuGraph, PyGraphistry

## Version

**llcuda v2.2.0** - CUDA12 Inference Backend for Unsloth

Released: January 2025

## License

MIT License - Copyright © 2024-2026 Waqas Muhammad
