# Graphistry Integration

Knowledge graph visualization with RAPIDS on GPU 1.

## Overview

Run Graphistry on GPU 1 while LLM runs on GPU 0.

## Basic Usage

```python
import os

# LLM on GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from llcuda.server import ServerManager
server = ServerManager()

# Graphistry on GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import graphistry
graphistry.register(api=3)
```

## Split-GPU Workflow

```
GPU 0: llama-server (LLM)
  ↓ Extract entities
GPU 1: RAPIDS + Graphistry
  → Visualize graphs
```

## Examples

See [Split-GPU Tutorial](../tutorials/06-split-gpu-graphistry.md)
