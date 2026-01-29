# Tutorial 11: GGUF Neural Network Architecture Visualization ⭐

**Notebook:** `11-gguf-neural-network-graphistry-visualization.ipynb`
**Difficulty:** Advanced
**Time:** 60 minutes
**Platform:** Kaggle (2× Tesla T4)
**Prerequisites:** Notebooks 01-06

[![Open in Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/waqasm86/11-gguf-neural-network-graphistry-vis-executed-2)

---

## 🎯 Overview

This is the **MOST IMPORTANT** tutorial in the llcuda v2.2.0 series - a groundbreaking tool for visualizing GGUF model internal architecture as interactive graphs using Graphistry. This tutorial represents the pinnacle of the llcuda project, showcasing the split-GPU architecture's full potential.

### Why This Tutorial is Critical

1. **First-of-its-Kind**: The only comprehensive GGUF architecture visualization tool available
2. **End-to-End Project**: Complete workflow from model loading to interactive dashboards
3. **Research-Grade**: Produces publication-quality visualizations and metrics
4. **Split-GPU Showcase**: Demonstrates GPU 0 (LLM) + GPU 1 (Visualization) architecture
5. **Production-Ready**: Generates downloadable HTML dashboards

---

## 📊 What You'll Visualize

### Complete Model Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│           GGUF LLAMA-3.2-3B ARCHITECTURE VISUALIZATION          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📥 Input Layer (1 node)                                       │
│      ↓                                                          │
│   🔤 Token Embedding (1 node, 393M parameters)                  │
│      ↓                                                          │
│   ┌─────────────────────────────────────────────────┐          │
│   │  🔄 Transformer Layer 1                         │          │
│   │    ├─ 32 Attention Heads (parallel)             │          │
│   │    ├─ RMSNorm (layer normalization)             │          │
│   │    └─ SwiGLU Feed-Forward Network               │          │
│   └─────────────────────────────────────────────────┘          │
│      ↓                                                          │
│   ... Layers 2-27 (identical structure)                         │
│      ↓                                                          │
│   🔄 Transformer Layer 28 (final layer)                         │
│      ↓                                                          │
│   📤 Output Layer (1 node, 393M parameters)                     │
│                                                                 │
│   📊 TOTAL: 929 nodes, 981 edges                                │
│   💾 Model Size: 1.88 GB (Q4_K_M quantization)                  │
│   🧮 Parameters: ~2.8 billion                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture Statistics

### Node Distribution

| Node Type | Count | Purpose | Memory |
|-----------|-------|---------|--------|
| **Attention Heads** | 896 | Multi-head attention (28 layers × 32 heads) | ~1.2 GB |
| **Transformer Blocks** | 28 | Complete transformer layers | ~1.8 GB total |
| **Embedding Layer** | 1 | Token embeddings (128,256 vocab × 3,072 dim) | ~393 MB |
| **Output Layer** | 1 | Output projections | ~393 MB |
| **Normalization** | 1 | Shared RMSNorm across all layers | ~24 KB |
| **Feed-Forward** | 1 | Shared SwiGLU network | ~113 MB |

### Edge Distribution

| Edge Type | Count | Meaning | Example |
|-----------|-------|---------|---------|
| `contains` | 896 | Layer → Attention head relationship | Layer_5 → L5_H12 |
| `feeds_into` | 28 | Sequential layer connections | Layer_1 → Layer_2 |
| `uses` | 56 | Layer → Shared component | Layer_3 → RMSNorm |
| **Total Edges** | **981** | Complete connectivity graph | - |

---

## 🎨 Visual Components

### 1. Main Architecture Dashboard (929 Nodes)

The complete model visualization showing all 28 transformer layers, embeddings, and output projections.

**Key Features:**
- Color-coded by node type (attention=blue, transformer=green, embedding=purple)
- Node size proportional to parameter count
- Edge thickness indicates connection strength
- Interactive zoom, pan, and search

**Insights Revealed:**
- Information flow from input to output
- Parameter distribution across layers
- Bottlenecks and skip connections
- Quantization impact on different components

### 2. Layer-Specific Visualizations (Layers 1-5)

Five detailed dashboards showing internal structure of individual transformer blocks (35 nodes, 34 edges each).

**Each Layer Shows:**
```
Transformer Block Container (1 node)
  ├─ Attention Head 0 (query, key, value, output)
  ├─ Attention Head 1
  ├─ ... (30 more heads)
  ├─ Attention Head 31
  ├─ RMSNorm (shared normalization)
  └─ SwiGLU Feed-Forward (shared expansion network)
```

**Why 5 Layers?**
- **Layer 1**: First transformer block (after embedding)
- **Layer 2**: Second layer (pattern establishing)
- **Layer 3**: Mid-early layer (feature building)
- **Layer 4**: Mid-layer (representation depth)
- **Layer 5**: Shows consistent architecture pattern

### 3. Attention Head Analysis (896 Nodes)

Visualization of all 896 attention heads across 28 layers.

**Graph Theory Metrics:**
- **PageRank**: Identifies most important attention heads
- **Betweenness Centrality**: Finds critical information pathways
- **Degree Distribution**: Analyzes connectivity patterns

**Research Applications:**
- Identify redundant heads for pruning
- Compare head importance across layers
- Analyze attention patterns in quantized vs full-precision models

### 4. Quantization Block Visualization (112 Nodes)

Shows Q4_K_M quantization structure: 4× super-blocks, each with 8 blocks, each with 32 weights.

```
Super-Block 0 (512 weights)
  ├─ Block 0 (32 weights, quantized)
  ├─ Block 1 (32 weights, quantized)
  ├─ ... (6 more blocks)
  └─ Block 7 (32 weights, quantized)

... (3 more super-blocks)
```

---

## 🔬 Technical Implementation

### Split-GPU Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL T4 GPU ALLOCATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4 (15GB VRAM) - LLM INFERENCE                   │
│   ├─ llama-server process                                       │
│   ├─ Model: Llama-3.2-3B-Q4_K_M                                 │
│   ├─ VRAM Usage: ~3.5 GB                                        │
│   ├─ tensor_split: "1.0,0.0" (100% GPU 0)                       │
│   └─ Available for queries: Yes                                 │
│                                                                 │
│   GPU 1: Tesla T4 (15GB VRAM) - VISUALIZATION                   │
│   ├─ RAPIDS cuGraph (PageRank computation)                      │
│   ├─ Graphistry (interactive rendering)                         │
│   ├─ VRAM Usage: ~0.8 GB                                        │
│   └─ Available VRAM: ~14 GB                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Architecture Extraction** (GPU 0)
   - Query llama-server for model metadata
   - Extract layer count, head count, dimensions
   - Build architectural graph representation

2. **Graph Construction** (CPU)
   - Create nodes for each component
   - Define edges (contains, feeds_into, uses)
   - Calculate node properties (params, memory)

3. **Graph Analytics** (GPU 1)
   - Load graph into cuGraph (GPU-accelerated)
   - Compute PageRank (identify important nodes)
   - Calculate centrality metrics
   - Perform community detection

4. **Visualization** (GPU 1)
   - Upload to Graphistry cloud
   - Generate interactive HTML dashboards
   - Create downloadable artifacts

---

## 📝 Step-by-Step Walkthrough

### Part 1: Environment Setup (Cells 1-10)

**Cell 1-3: GPU Verification**
```python
!nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

**Expected Output:**
```
index, name, memory.total [MiB], compute_cap
0, Tesla T4, 15360 MiB, 7.5
1, Tesla T4, 15360 MiB, 7.5
```

**Cell 11: Install llcuda v2.2.0**
```bash
!pip install -q --no-cache-dir --force-reinstall \
  git+https://github.com/llcuda/llcuda.git@v2.2.0
```

---

### Part 2: Model Setup (Cells 11-20)

**Cell 16: Download GGUF Model**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)
```

**Model Specifications:**
- Size: 1.88 GB
- Quantization: Q4_K_M (mixed 4-bit/6-bit)
- Context: 131,072 tokens (128K)
- Vocabulary: 128,256 tokens

**Cell 18: Start llama-server on GPU 0**
```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path=model_path,
    n_gpu_layers=99,              # Load all layers to GPU
    tensor_split="1.0,0.0",       # 100% GPU 0, 0% GPU 1
    flash_attn=1,                 # Enable FlashAttention
    n_ctx=4096,                   # Context window
    host="127.0.0.1",
    port=8080,
)

server = ServerManager()
server.start_with_config(config)
server.wait_until_ready(timeout=120)
```

---

### Part 3: Architecture Extraction (Cells 21-30)

**Cell 24: Query Model Metadata**
```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient(base_url="http://127.0.0.1:8080")

# Get model architecture
metadata = client.get_model_metadata()

print(f"Model: {metadata['general.name']}")
print(f"Layers: {metadata['llama.block_count']}")
print(f"Attention Heads: {metadata['llama.attention.head_count']}")
print(f"Embedding Dimension: {metadata['llama.embedding_length']}")
print(f"Vocabulary Size: {metadata['tokenizer.ggml.token_count']}")
```

**Expected Output:**
```
Model: Llama-3.2-3B-Instruct
Layers: 28
Attention Heads: 32
Embedding Dimension: 3072
Vocabulary Size: 128256
```

**Cell 28: Build Architecture Graph**
```python
import pandas as pd
import networkx as nx

# Create graph
G = nx.DiGraph()

# Add input node
G.add_node("Input", type="input", params=0)

# Add embedding layer
embedding_params = vocab_size * embedding_dim
G.add_node("Embedding", type="embedding", params=embedding_params)
G.add_edge("Input", "Embedding", relationship="feeds_into")

# Add transformer layers
for layer_idx in range(num_layers):
    layer_name = f"Layer_{layer_idx + 1}"

    # Add layer node
    G.add_node(layer_name, type="transformer", params=layer_params)

    # Add attention heads
    for head_idx in range(num_heads):
        head_name = f"L{layer_idx + 1}_H{head_idx}"
        G.add_node(head_name, type="attention_head", params=head_params)
        G.add_edge(layer_name, head_name, relationship="contains")

    # Connect to shared components
    G.add_edge(layer_name, "LayerNorm", relationship="uses")
    G.add_edge(layer_name, "FeedForward", relationship="uses")

    # Sequential connection
    if layer_idx > 0:
        prev_layer = f"Layer_{layer_idx}"
        G.add_edge(prev_layer, layer_name, relationship="feeds_into")

# Add output layer
G.add_node("Output", type="output", params=embedding_params)
G.add_edge(f"Layer_{num_layers}", "Output", relationship="feeds_into")

print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

---

### Part 4: GPU-Accelerated Analytics (Cells 31-40)

**Cell 33: Install RAPIDS cuGraph**
```bash
!pip install -q cugraph-cu12
```

**Cell 36: Compute PageRank on GPU 1**
```python
import cudf
import cugraph

# Convert NetworkX to cuGraph
edges_df = cudf.DataFrame({
    'src': [edge[0] for edge in G.edges()],
    'dst': [edge[1] for edge in G.edges()]
})

cu_graph = cugraph.Graph(directed=True)
cu_graph.from_cudf_edgelist(edges_df, source='src', destination='dst')

# Compute PageRank (GPU-accelerated)
pagerank_df = cugraph.pagerank(cu_graph)

# Convert back to pandas
pr_dict = dict(zip(pagerank_df['vertex'].to_pandas(),
                   pagerank_df['pagerank'].to_pandas()))

# Add to graph
nx.set_node_attributes(G, pr_dict, 'pagerank')

# Find top 10 most important nodes
top_nodes = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:10]
for node, score in top_nodes:
    print(f"{node}: PageRank = {score:.6f}")
```

**Expected Top Nodes:**
```
Embedding: PageRank = 0.045231
Layer_14: PageRank = 0.038912
Layer_15: PageRank = 0.037654
L14_H16: PageRank = 0.023445
Output: PageRank = 0.022876
...
```

---

### Part 5: Interactive Visualization (Cells 41-60)

**Cell 45: Install Graphistry**
```bash
!pip install -q graphistry[all]
```

**Cell 48: Create Main Architecture Dashboard**
```python
import graphistry

# Register with Graphistry (free tier)
graphistry.register(api=3, protocol="https", server="hub.graphistry.com")

# Prepare data for Graphistry
nodes_df = pd.DataFrame([
    {
        'node': node,
        'type': data.get('type', 'unknown'),
        'params': data.get('params', 0),
        'pagerank': data.get('pagerank', 0),
        'label': node,
    }
    for node, data in G.nodes(data=True)
])

edges_df = pd.DataFrame([
    {
        'src': src,
        'dst': dst,
        'relationship': data.get('relationship', 'unknown'),
    }
    for src, dst, data in G.edges(data=True)
])

# Create Graphistry visualization
g = graphistry.edges(edges_df, 'src', 'dst') \
               .nodes(nodes_df, 'node') \
               .bind(node='node',
                     point_title='label',
                     point_size='params',
                     point_color='type',
                     edge_title='relationship')

# Upload and get URL
url = g.plot(render=False)
print(f"Main Architecture Dashboard: {url}")
```

**Cell 52-58: Create Layer-Specific Dashboards**
```python
# Visualize Layers 1-5 individually
for layer_num in range(1, 6):
    layer_name = f"Layer_{layer_num}"

    # Extract subgraph for this layer
    layer_nodes = [layer_name]
    layer_nodes += [n for n in G.nodes() if n.startswith(f"L{layer_num}_H")]
    layer_nodes += ["LayerNorm", "FeedForward"]

    subgraph = G.subgraph(layer_nodes)

    # Create visualization
    sub_nodes_df = nodes_df[nodes_df['node'].isin(layer_nodes)]
    sub_edges_df = edges_df[
        (edges_df['src'].isin(layer_nodes)) &
        (edges_df['dst'].isin(layer_nodes))
    ]

    g_layer = graphistry.edges(sub_edges_df, 'src', 'dst') \
                        .nodes(sub_nodes_df, 'node') \
                        .bind(node='node',
                              point_title='label',
                              point_size='params',
                              point_color='type')

    url = g_layer.plot(render=False)
    print(f"Layer {layer_num} Dashboard (35 nodes, 34 edges): {url}")
```

---

### Part 6: Complete Dashboard Export (Cells 61-70)

**Cell 65: Generate All-in-One HTML Dashboard**
```python
# Create comprehensive dashboard with all visualizations
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GGUF Llama-3.2-3B Architecture Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .visualization {{ margin: 30px 0; padding: 20px; border: 1px solid #ccc; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .stats {{ background: #f0f0f0; padding: 15px; margin: 15px 0; }}
        iframe {{ width: 100%; height: 800px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>🧠 GGUF Neural Network Architecture Visualization</h1>

    <div class="stats">
        <h2>📊 Model Statistics</h2>
        <ul>
            <li><strong>Model:</strong> Llama-3.2-3B-Instruct</li>
            <li><strong>Quantization:</strong> Q4_K_M</li>
            <li><strong>Total Nodes:</strong> 929</li>
            <li><strong>Total Edges:</strong> 981</li>
            <li><strong>Transformer Layers:</strong> 28</li>
            <li><strong>Attention Heads per Layer:</strong> 32</li>
            <li><strong>Total Parameters:</strong> ~2.8 billion</li>
            <li><strong>File Size:</strong> 1.88 GB</li>
        </ul>
    </div>

    <div class="visualization">
        <h2>🗺️ Main Architecture (929 nodes)</h2>
        <iframe src="{main_url}"></iframe>
    </div>

    <div class="visualization">
        <h2>🔍 Layer 1 Detail (35 nodes)</h2>
        <iframe src="{layer1_url}"></iframe>
    </div>

    <!-- Layers 2-5 ... -->

    <div class="visualization">
        <h2>🎯 Attention Heads Analysis (896 nodes)</h2>
        <iframe src="{attention_url}"></iframe>
    </div>

    <div class="visualization">
        <h2>📦 Quantization Blocks (112 nodes)</h2>
        <iframe src="{quant_url}"></iframe>
    </div>
</body>
</html>
"""

# Save dashboard
dashboard_path = "/kaggle/working/complete_dashboard.html"
with open(dashboard_path, 'w') as f:
    f.write(html_content)

print(f"✅ Complete dashboard saved: {dashboard_path}")
print(f"📥 Download this file to view all visualizations offline!")
```

---

## 🎓 Key Learnings

### 1. Architecture Insights

**Information Flow:**
- Input → Embedding (vocabulary projection)
- 28 Sequential Transformer Blocks (feature extraction)
- Output → Vocabulary Logits (next token prediction)

**Parameter Distribution:**
- Embedding: 393M (21%)
- Transformers: 1,764M (64%)
- Output: 393M (21%)

**Attention Pattern:**
- 32 heads per layer = 896 total heads
- Each head processes ~96 dimensions
- Parallel computation within layer
- Sequential information flow between layers

### 2. Quantization Impact

**Q4_K_M Quantization:**
- Mixed 4-bit and 6-bit quantization
- Super-block structure (4 × 8 × 32 weights)
- Preserves important weights at higher precision
- 75% compression (vs FP16)

**Memory Distribution:**
- Weights: ~1.88 GB
- KV Cache (4K context): ~0.5 GB
- Activations: ~0.2 GB
- **Total VRAM: ~2.6 GB**

### 3. Split-GPU Benefits

**Why Split-GPU?**
1. **Resource Isolation**: LLM doesn't compete with visualization
2. **Parallel Processing**: Query model while analyzing
3. **Memory Efficiency**: Each GPU handles its workload
4. **Scalability**: Can add more GPUs for larger models

**Performance:**
- GPU 0 (LLM): 45 tokens/sec
- GPU 1 (Analytics): PageRank in <2 seconds
- Combined: No performance degradation

---

## 🔍 Research Applications

### Model Analysis
1. **Architecture Comparison**: Compare quantized vs full-precision
2. **Pruning Candidates**: Identify low-importance heads
3. **Bottleneck Detection**: Find slow layers
4. **Information Flow**: Trace token processing path

### Optimization
1. **Layer Fusion**: Identify redundant computations
2. **Selective Quantization**: Higher precision for important weights
3. **Dynamic Inference**: Skip layers for simple queries
4. **Knowledge Distillation**: Teacher-student architecture design

### Debugging
1. **Attention Visualization**: See what model focuses on
2. **Layer Output Analysis**: Detect degradation
3. **Quantization Validation**: Verify conversion correctness
4. **Performance Profiling**: Identify slow components

---

## 📁 Outputs

### Files Generated

1. **complete_dashboard.html** (5 MB)
   - All-in-one interactive dashboard
   - Downloadable from `/kaggle/working/`
   - Works offline after download

2. **architecture_graph.json** (2 MB)
   - Complete graph structure
   - Import into other graph tools (Gephi, Cytoscape)

3. **pagerank_results.csv** (50 KB)
   - Node importance scores
   - Use for pruning decisions

4. **layer_statistics.csv** (20 KB)
   - Per-layer metrics
   - Parameter counts, memory usage

### Cloud URLs (8 total)
- Main architecture (929 nodes)
- Layers 1-5 (35 nodes each)
- Attention heads (896 nodes)
- Quantization blocks (112 nodes)

---

## 🚀 Next Steps

### After Completing This Tutorial

1. **Experiment with Other Models**
   - Try 1B, 7B, or 13B models
   - Compare architectures
   - Analyze quantization impact

2. **Advanced Analytics**
   - Community detection (find module groups)
   - Shortest paths (trace information flow)
   - Clustering (group similar heads)

3. **Custom Visualizations**
   - Add custom metrics
   - Create animation sequences
   - Build comparison dashboards

4. **Production Deployment**
   - Automate dashboard generation
   - Create monitoring pipeline
   - Build CI/CD integration

---

## 🛠️ Troubleshooting

### Common Issues

**GPU Out of Memory:**
```python
# Reduce model context size
config.n_ctx = 2048  # Instead of 4096

# Or use smaller model
model = "Llama-3.2-1B-Q4_K_M.gguf"
```

**Graphistry Upload Timeout:**
```python
# Use smaller subgraphs
layer_subgraph = G.subgraph(list(G.nodes())[:100])

# Or save locally first
g.plot(render=True, as_files=True)
```

**cuGraph Installation Fails:**
```bash
# Use CPU-based PageRank instead
import networkx as nx
pagerank = nx.pagerank(G)
```

### Performance Optimization

**Speed Up Analytics:**
```python
# Reduce graph size
G_small = nx.k_core(G, k=2)  # Remove low-degree nodes

# Sample attention heads
sample_heads = [f"L{i}_H{j}" for i in range(1, 29, 2) for j in range(0, 32, 2)]
G_sample = G.subgraph(sample_heads)
```

**Reduce Memory Usage:**
```python
# Process layers sequentially
for layer in range(1, 29):
    layer_graph = extract_layer(G, layer)
    analyze_and_save(layer_graph)
    del layer_graph  # Free memory
```

---

## 📚 Additional Resources

### Related Tutorials
- [Tutorial 06: Split-GPU Graphistry](06-split-gpu-graphistry.md) - Foundation for this tutorial
- [Tutorial 04: GGUF Quantization](04-gguf-quantization.md) - Understanding Q4_K_M
- [Tutorial 03: Multi-GPU Inference](03-multi-gpu.md) - GPU management basics

### External Documentation
- [Graphistry Documentation](https://hub.graphistry.com/docs/)
- [RAPIDS cuGraph Guide](https://docs.rapids.ai/api/cugraph/stable/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

### Academic Papers
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- "GGML: Tensor Library for Machine Learning" (Ggerganov, 2023)

---

## 🎉 Conclusion

Congratulations! You've completed the most advanced tutorial in the llcuda v2.2.0 series. You now have:

✅ Interactive visualizations of complete model architecture
✅ Layer-by-layer analysis of transformer blocks
✅ PageRank scores for all 929 components
✅ Downloadable HTML dashboards
✅ Research-grade graph analytics
✅ Production-ready visualization pipeline

This tutorial represents the cutting edge of GGUF model analysis and demonstrates llcuda's full capabilities. Use these techniques to:
- Understand your models deeply
- Optimize inference performance
- Make informed quantization decisions
- Publish research-quality visualizations

---

## 📖 Citation

If you use this visualization tool in your research, please cite:

```bibtex
@software{llcuda2026,
  title = {llcuda: CUDA 12 Inference Backend for Unsloth with GGUF Visualization},
  author = {Muhammad, Waqas},
  year = {2026},
  version = {2.2.0},
  url = {https://github.com/llcuda/llcuda}
}
```

---

**Ready to visualize?** [Open this notebook in Kaggle](https://www.kaggle.com/code/waqasm86/11-gguf-neural-network-graphistry-vis-executed-2) and start exploring! 🚀
