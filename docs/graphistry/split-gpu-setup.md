# Split-GPU Setup

Configure LLM on GPU 0 and Graphistry on GPU 1.

## Architecture

```
GPU 0: llama-server (15GB)
  ↓ Extract knowledge graphs
GPU 1: RAPIDS + Graphistry (15GB)
  → Visualize millions of nodes/edges
```

## Setup GPU 0 (LLM)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(model_path="model.gguf", n_gpu_layers=99)
server = ServerManager()
server.start_with_config(config)
```

## Setup GPU 1 (Graphistry)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import graphistry
graphistry.register(api=3, protocol="https", server="hub.graphistry.com")
```

## Workflow

1. Run LLM inference on GPU 0
2. Extract entities/relations from output
3. Build graph on GPU 1 with cuDF
4. Visualize with Graphistry

## See Also

- [Split-GPU Tutorial](../tutorials/06-split-gpu-graphistry.md)
- [Knowledge Graphs](../graphistry/knowledge-graphs.md)
