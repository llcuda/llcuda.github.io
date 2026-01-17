# Split-GPU with Graphistry

LLM on GPU 0 + RAPIDS/Graphistry visualization on GPU 1.

**Level**: Intermediate | **Time**: 30 minutes | **VRAM Required**: GPU0: 5-10 GB, GPU1: 2-8 GB

---

## Split-GPU Architecture

```
GPU 0: llama-server (LLM inference)
  ↓ Extract knowledge graphs
GPU 1: RAPIDS cuDF/cuGraph + Graphistry (visualization)
```

## Configure LLM on GPU 0

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    flash_attn=True
)

server = ServerManager()
server.start_with_config(config)
```

## Configure Graphistry on GPU 1

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cudf
import graphistry

graphistry.register(api=3, protocol="https", server="hub.graphistry.com")
```

## Knowledge Graph Workflow

```python
# 1. Extract entities from LLM
from llcuda.api.client import LlamaCppClient

client = LlamaCppClient()
response = client.create_chat_completion(
    messages=[{"role": "user", "content": "Extract entities from: ..."}]
)

# 2. Build graph on GPU 1
edges_df = cudf.DataFrame(...)
g = graphistry.edges(edges_df)
g.plot()
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/06-split-gpu-graphistry-llcuda-v2-2-0)
