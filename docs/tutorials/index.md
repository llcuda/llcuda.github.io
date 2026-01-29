# Tutorial Notebooks

Complete tutorial series for llcuda v2.2.0 on Kaggle dual T4 - **11 comprehensive tutorials** from beginner to advanced.

## Core Tutorials (1-10)

| # | Notebook | Open in Kaggle | Description | Time |
|---|----------|----------------|-------------|------|
| 01 | [Quick Start](01-quickstart.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/01-quickstart-llcuda-v2-2-0) | 5-minute introduction | 5 min |
| 02 | [Server Setup](02-server-setup.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/02-llama-server-setup-llcuda-v2-2-0) | Server configuration | 15 min |
| 03 | [Multi-GPU](03-multi-gpu.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/03-multi-gpu-inference-llcuda-v2-2-0) | Dual T4 tensor-split | 20 min |
| 04 | [GGUF Quantization](04-gguf-quantization.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/04-gguf-quantization-llcuda-v2-2-0) | K-quants, I-quants | 20 min |
| 05 | [Unsloth Integration](05-unsloth-integration.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/05-unsloth-integration-llcuda-v2-2-0) | Fine-tune → Deploy | 30 min |
| 06 | [Split-GPU + Graphistry](06-split-gpu-graphistry.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/06-split-gpu-graphistry-llcuda-v2-2-0) | LLM + Visualization | 30 min |
| 07 | [OpenAI API](07-openai-api.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/07-openai-api-client-llcuda-v2-2-0) | OpenAI SDK | 15 min |
| 08 | [NCCL + PyTorch](08-nccl-pytorch.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/08-nccl-pytorch-llcuda-v2-2-0) | Distributed PyTorch | 25 min |
| 09 | [Large Models (70B)](09-large-models.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/09-large-models-kaggle-llcuda-v2-2-0) | 70B on dual T4 | 30 min |
| 10 | [Complete Workflow](10-complete-workflow.md) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/10-complete-workflow-llcuda-v2-2-0) | End-to-end | 45 min |

## ⭐ Advanced Visualization (Tutorial 11) - MOST IMPORTANT

| # | Notebook | Open in Kaggle | Description | Time |
|---|----------|----------------|-------------|------|
| 11 | [**GGUF Neural Network Visualization**](11-gguf-neural-network-visualization.md) ⭐ | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/11-gguf-neural-network-graphistry-vis-executed-2) | **Complete model architecture as interactive graphs** | 60 min |

**Why Tutorial 11 is Critical:**
- 🏆 **First-of-its-kind**: Only comprehensive GGUF visualization tool
- 📊 **929 nodes, 981 edges**: Complete Llama-3.2-3B architecture
- 🎨 **Interactive dashboards**: 8 Graphistry cloud visualizations
- 🔬 **Research-grade**: PageRank, centrality, community detection
- 🖥️ **Split-GPU showcase**: LLM (GPU 0) + Analytics (GPU 1)
- 📥 **Downloadable**: HTML dashboards for offline viewing

**What You'll Visualize:**
- Complete 28-layer transformer architecture
- 896 attention heads across all layers
- Layer-by-layer breakdowns (35 nodes each)
- Q4_K_M quantization block structure
- Information flow through the network

## Learning Paths

### Beginner (1 hour)
**Start here** if you're new to llcuda:
```
01 → 02 → 03
Quick Start → Server Setup → Multi-GPU
```

### Intermediate (3 hours)
**Full fundamentals** with deployment:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 10
All basics through complete workflow
```

### Advanced (2 hours)
**Multi-GPU focus** for large models:
```
01 → 03 → 08 → 09
Quick Start → Multi-GPU → NCCL → 70B Models
```

### Visualization & Research (2.5 hours) ⭐ **RECOMMENDED**
**Complete architecture analysis** with Graphistry:
```
01 → 03 → 04 → 06 → 11
Quick Start → Multi-GPU → GGUF → Split-GPU → Architecture Visualization
```

### Complete Master (6 hours)
**Everything** from basics to advanced visualization:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11
All tutorials in order
```
