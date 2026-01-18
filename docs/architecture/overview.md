# Architecture Overview

llcuda v2.2.0 architecture for Kaggle dual T4.

## System Architecture

```
┌─────────────────────────────────────────┐
│         llcuda v2.2.0 Stack             │
├─────────────────────────────────────────┤
│  Python API (llcuda.api.*)              │
│  ├─ client.py (OpenAI-compatible)       │
│  ├─ multigpu.py (Dual T4 config)        │
│  ├─ gguf.py (Quantization tools)        │
│  └─ nccl.py (PyTorch distributed)       │
├─────────────────────────────────────────┤
│  Server Manager (llcuda.server)         │
│  └─ Lifecycle management                │
├─────────────────────────────────────────┤
│  llama.cpp Server (C++/CUDA)            │
│  ├─ Build 7760 (commit 388ce82)         │
│  ├─ OpenAI API endpoints                │
│  └─ Native CUDA tensor-split            │
├─────────────────────────────────────────┤
│  CUDA 12.5 / cuBLAS                     │
│  ├─ FlashAttention kernels              │
│  ├─ Tensor Core optimization            │
│  └─ SM 7.5 (Turing)                     │
├─────────────────────────────────────────┤
│  Kaggle Dual T4 (30GB VRAM)             │
│  ├─ GPU 0: LLM Inference                │
│  └─ GPU 1: Graphistry/RAPIDS            │
└─────────────────────────────────────────┘
```

## Key Components

- **Python APIs**: High-level interfaces
- **Server Manager**: Process lifecycle  
- **llama.cpp**: CUDA inference engine
- **Split-GPU**: Dual GPU coordination

See:
- [Split-GPU Design](split-gpu.md)
- [GPU0 - LLM](gpu0-llm.md)
- [GPU1 - Graphistry](gpu1-graphistry.md)
