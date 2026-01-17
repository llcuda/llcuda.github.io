# GPU 1 - Graphistry

Use GPU 1 for RAPIDS + Graphistry visualization.

## Setup

```python
import graphistry
import cudf

# Configure Graphistry for GPU 1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Register Graphistry
graphistry.register(
    api=3,
    protocol="https",
    server="hub.graphistry.com",
    personal_key_id="YOUR_KEY"
)
```

## Workflow

1. **Extract from LLM** (GPU 0)
   ```python
   # Get entities from LLM
   entities = llm_client.extract_entities(text)
   ```

2. **Build Graph** (GPU 1)
   ```python
   # Create graph with cuDF
   nodes_df = cudf.DataFrame(entities)
   edges_df = cudf.DataFrame(relationships)
   ```

3. **Visualize** (GPU 1)
   ```python
   # Render with Graphistry
   g = graphistry.edges(edges_df).nodes(nodes_df)
   g.plot()
   ```

## VRAM Usage

- **cuDF**: 1-3 GB
- **cuGraph**: 2-5 GB
- **Graphistry**: 1-4 GB
- **Total**: 4-12 GB on GPU 1
