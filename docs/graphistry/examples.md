# Graphistry Examples

Example workflows for LLM + Graphistry visualization.

## Extract Entities from LLM

```python
from llcuda.api.client import LlamaCppClient

client = LlamaCppClient()
response = client.create_chat_completion(
    messages=[{
        "role": "user",
        "content": "Extract entities and relationships from: ..."
    }]
)

# Parse LLM output into entities and edges
```

## Build Graph on GPU 1

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cudf
import graphistry

# Create edges dataframe
edges_df = cudf.DataFrame({
    "src": ["Alice", "Bob", "Charlie"],
    "dst": ["Bob", "Charlie", "Alice"],
    "relation": ["knows", "works_with", "manages"]
})

# Visualize
g = graphistry.edges(edges_df, "src", "dst")
g.plot()
```

## Full Workflow

See [Split-GPU Tutorial](../tutorials/06-split-gpu-graphistry.md)
