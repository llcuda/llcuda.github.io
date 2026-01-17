# RAPIDS Integration

Use RAPIDS cuDF/cuGraph on GPU 1 alongside LLM on GPU 0.

## Overview

RAPIDS provides GPU-accelerated dataframes and graph analytics.

## Installation

```bash
pip install cudf-cu12 cugraph-cu12 graphistry
```

## Basic Usage

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

import cudf
import cugraph

# Create graph
df = cudf.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 0]})
G = cugraph.Graph()
G.from_cudf_edgelist(df, source="src", destination="dst")
```

## With LLM

```python
# GPU 0: LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from llcuda.server import ServerManager
server = ServerManager()

# GPU 1: RAPIDS
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cudf
# Process data on GPU 1
```

## See Also

- [Split-GPU Tutorial](../tutorials/06-split-gpu-graphistry.md)
- [Graphistry Examples](examples.md)
