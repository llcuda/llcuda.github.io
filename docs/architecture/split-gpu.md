# Split-GPU Architecture

Run LLM on GPU 0 + Graphistry on GPU 1.

## Architecture

```
GPU 0 (T4 - 15GB)          GPU 1 (T4 - 15GB)
┌──────────────────┐      ┌──────────────────┐
│  llama-server    │      │  RAPIDS cuDF     │
│  GGUF Model      │ ────>│  cuGraph         │
│  LLM Inference   │      │  Graphistry[ai]  │
│  ~5-12GB VRAM    │      │  Network Viz     │
└──────────────────┘      └──────────────────┘
```

## Configuration

```python
from llcuda.graphistry import SplitGPUConfig
import os

# Assign GPUs
config = SplitGPUConfig(
    llm_gpu=0,      # GPU 0 for LLM
    graph_gpu=1     # GPU 1 for Graphistry
)

# Set CUDA device for llama-server
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Start llama-server on GPU 0
# (Graphistry will use GPU 1)
```

## Use Cases

1. **Knowledge Graph Extraction**
   - LLM generates entities/relationships
   - Graphistry visualizes graphs

2. **Interactive Analysis**
   - LLM answers questions
   - Graphistry shows data patterns

3. **Multi-Modal Workflows**
   - Text generation (GPU 0)
   - Graph analytics (GPU 1)
