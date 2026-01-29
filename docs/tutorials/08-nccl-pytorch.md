# Document Network Analysis with Graphistry

Analyze document similarity and topic clustering using GPU-accelerated graph analytics.

**Level**: Advanced | **Time**: 35 minutes | **VRAM Required**: GPU 0: 6-10 GB, GPU 1: 3-5 GB

---

---

## Overview

This notebook demonstrates how to analyze document similarity and topic clustering using GPU-accelerated graph analytics with RAPIDS cuGraph and interactive visualization with Graphistry.

## Key Features

- **Document embedding** generation via LLM
- **Similarity network** construction (cosine similarity)
- **Community detection** using RAPIDS cuGraph
- **Topic clustering** with GPU-accelerated algorithms
- **Interactive visualization** with Graphistry
- **Dual-GPU workflow** (embeddings on GPU 0, analytics on GPU 1)

## Key Algorithms

- **Louvain community detection** - Find document clusters
- **PageRank** - Identify influential documents
- **Betweenness centrality** - Find bridge documents
- **K-core decomposition** - Extract dense subnetworks

## Applications

- Research paper citation networks
- News article topic analysis
- Corporate document organization
- Social media content clustering

## Workflow

```python
# 1. Generate embeddings
embeddings = get_embeddings_from_llm(documents)

# 2. Build similarity graph
similarity_matrix = cosine_similarity(embeddings)
graph = build_graph_from_similarity(similarity_matrix, threshold=0.7)

# 3. GPU analytics with cuGraph
communities = cugraph.louvain(graph)
pagerank = cugraph.pagerank(graph)

# 4. Visualize
g = graphistry.nodes(docs_df).edges(edges_df)
g.plot()
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/08-document-network-analysis-graphistry-llcuda-v2-2-0)
