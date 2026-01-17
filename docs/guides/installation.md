# Installation Guide

Complete installation guide for llcuda v2.2.0 on Kaggle dual T4 GPUs.

---

## Requirements

### Hardware

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA Tesla T4 (Kaggle 2× T4) |
| **VRAM** | 15GB (single T4) or 30GB (dual T4) |
| **RAM** | 16GB+ recommended |

### Software

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.11 or higher |
| **CUDA** | 12.x runtime |
| **OS** | Linux (Ubuntu 20.04+, Kaggle) |
| **pip** | 23.0+ |

---

## Kaggle Installation (Recommended)

### Step 1: Configure Notebook Settings

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Create new notebook
3. **Settings → Accelerator → GPU T4 × 2** ✅
4. **Settings → Internet → On** ✅

### Step 2: Install llcuda

```bash
# Install from GitHub v2.2.0
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

### Step 3: Verify Installation

```python
import llcuda
from llcuda.api.multigpu import detect_gpus, print_gpu_info

# Check version
print(f"llcuda version: {llcuda.__version__}")  # 2.2.0

# Verify dual T4 setup
gpus = detect_gpus()
print(f"Detected {len(gpus)} GPUs")
print_gpu_info()
```

**Expected output:**
```
llcuda version: 2.2.0
Detected 2 GPUs

GPU 0: Tesla T4
  Memory: 15.0 / 15.0 GB
  Compute Capability: 7.5

GPU 1: Tesla T4
  Memory: 15.0 / 15.0 GB
  Compute Capability: 7.5
```

---

## Binary Download

On first import, llcuda automatically downloads CUDA binaries:

- **Size**: 961 MB
- **Source**: [GitHub Releases v2.2.0](https://github.com/llcuda/llcuda/releases/tag/v2.2.0)
- **SHA256**: Automatically verified
- **Cache**: `~/.cache/llcuda/`

---

## Next Steps

- [Quick Start](quickstart.md)
- [Kaggle Setup](kaggle-setup.md)
- [First Steps](first-steps.md)
