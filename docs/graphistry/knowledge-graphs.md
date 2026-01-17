# Knowledge Graph Extraction

Extract knowledge graphs from LLM outputs.

## Workflow

### 1. Generate Text (GPU 0)
```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient()
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": "Extract entities and relationships from: ..."
    }]
)

text = response.choices[0].message.content
```

### 2. Parse Entities
```python
import json

# Parse LLM output
data = json.loads(text)
entities = data['entities']
relationships = data['relationships']
```

### 3. Build Graph (GPU 1)
```python
import cudf

nodes_df = cudf.DataFrame(entities)
edges_df = cudf.DataFrame(relationships)
```

### 4. Visualize (GPU 1)
```python
import graphistry

g = graphistry.edges(edges_df).nodes(nodes_df)
g.plot()
```

## Use Cases

- Document analysis
- Semantic networks
- Entity relationship mapping
- Knowledge base visualization
