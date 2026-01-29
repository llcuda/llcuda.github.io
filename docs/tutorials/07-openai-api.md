# Knowledge Graph Extraction with Graphistry

Extract knowledge graphs from unstructured text using LLM-powered entity recognition and visualize with Graphistry.

**Level**: Advanced | **Time**: 30 minutes | **VRAM Required**: GPU 0: 5-8 GB, GPU 1: 2-4 GB

---

---

## Overview

This notebook demonstrates how to extract knowledge graphs from unstructured text using LLM-powered entity recognition and visualize them with Graphistry on a split-GPU architecture.

## Key Concepts

- **LLM-based entity extraction** from documents
- **Relationship detection** between entities
- **Graph construction** with nodes (entities) and edges (relationships)
- **Graphistry visualization** with interactive exploration
- **GPU acceleration** using RAPIDS for large graphs
- **Split-GPU architecture** (LLM on GPU 0, Graphistry on GPU 1)

## Use Cases

- Academic paper analysis
- Legal document processing
- News article relationship mapping
- Scientific literature mining

## Workflow

```python
# 1. Extract entities and relationships using LLM
response = client.chat.create(
    messages=[{"role": "user", "content": f"Extract entities from: {text}"}]
)

# 2. Build graph
entities_df = pd.DataFrame(entities)
relationships_df = pd.DataFrame(relationships)

# 3. Visualize with Graphistry
g = graphistry.bind(source='from', destination='to', node='entity')
g.edges(relationships_df).nodes(entities_df).plot()
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/07-knowledge-graph-extraction-graphistry-v2-2-0)
