# Graphistry Integration

GPU-accelerated graph visualization with llcuda.

## What is Graphistry?

PyGraphistry provides:
- **GPU-accelerated** graph rendering
- **Millions** of nodes/edges
- **Interactive** exploration
- **RAPIDS** integration (cuDF, cuGraph)

## Split-GPU Architecture

```
GPU 0: llcuda (LLM)  →  GPU 1: Graphistry (Viz)
```

## Quick Start

```python
import graphistry
import cudf

# Configure for GPU 1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Register Graphistry
graphistry.register(
    api=3,
    protocol="https",
    server="hub.graphistry.com",
    personal_key_id="YOUR_KEY"
)

# Create graph
nodes = cudf.DataFrame({
    'id': [1, 2, 3],
    'label': ['A', 'B', 'C']
})

edges = cudf.DataFrame({
    'src': [1, 2],
    'dst': [2, 3]
})

# Visualize
g = graphistry.edges(edges, 'src', 'dst').nodes(nodes, 'id')
g.plot()
```

See:
- [Knowledge Graphs](knowledge-graphs.md)
- [RAPIDS Integration](rapids.md)
- [Examples](examples.md)
