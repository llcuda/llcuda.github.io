#!/usr/bin/env python3
"""
Generate complete llcuda v2.2.0 GitHub Pages documentation.
This script creates all documentation pages for the llcuda.github.io website.
"""

from pathlib import Path
import shutil

# Base directory
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"

# Ensure directory structure exists
for subdir in ["guides", "kaggle", "tutorials", "architecture", "api", "unsloth", "graphistry", "performance", "gguf"]:
    (DOCS_DIR / subdir).mkdir(parents=True, exist_ok=True)

print("✓ Directory structure created")

# Track created files
created_files = []

def create_file(path: Path, content: str):
    """Create a file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    created_files.append(str(path.relative_to(BASE_DIR)))
    print(f"  Created: {path.relative_to(BASE_DIR)}")

# ===================================================================
# GUIDES
# ===================================================================

create_file(DOCS_DIR / "guides/installation.md", """# Installation Guide

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
""")

create_file(DOCS_DIR / "guides/quickstart.md", """# Quick Start Guide

Get llcuda v2.2.0 running on Kaggle in 5 minutes!

---

## Step 1: Install (1 minute)

```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

---

## Step 2: Verify Dual T4 (30 seconds)

```python
from llcuda.api.multigpu import detect_gpus

gpus = detect_gpus()
print(f"✓ Detected {len(gpus)} GPUs")
for gpu in gpus:
    print(f"  GPU {gpu.id}: {gpu.name} ({gpu.memory_total_gb:.1f} GB)")
```

---

## Step 3: Start Server (2 minutes)

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

print("✓ Server running at http://localhost:8080")
```

---

## Step 4: Run Inference (1 minute)

```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

---

## Next Steps

- [Tutorial Notebooks](../tutorials/index.md)
- [Multi-GPU Guide](../kaggle/multi-gpu-inference.md)
- [API Reference](../api/overview.md)
""")

create_file(DOCS_DIR / "guides/first-steps.md", """# First Steps

Your first steps with llcuda v2.2.0 on Kaggle.

---

## 1. Load a Model

```python
from llcuda.server import ServerManager, ServerConfig

# Basic configuration
config = ServerConfig(
    model_path="/path/to/model.gguf",
    n_gpu_layers=99,  # Offload all to GPU
)

server = ServerManager()
server.start_with_config(config)
```

---

## 2. Make Your First Request

```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient()
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## 3. Explore Notebooks

Try the tutorial notebooks:
- [01 - Quick Start](../tutorials/01-quickstart.md)
- [02 - Server Setup](../tutorials/02-server-setup.md)
- [03 - Multi-GPU](../tutorials/03-multi-gpu.md)
""")

create_file(DOCS_DIR / "guides/kaggle-setup.md", """# Kaggle Setup Guide

Complete guide for setting up llcuda v2.2.0 on Kaggle with dual T4 GPUs.

---

## Prerequisites

- Kaggle account
- Phone verification (for GPU access)

---

## Step 1: Create Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click "New Notebook"
3. Choose "Notebook" type

---

## Step 2: Configure GPU

1. Click **Settings** (gear icon)
2. **Accelerator** → Select **GPU T4 x 2**
3. **Internet** → Toggle **On**
4. **Persistence** → Optional: Enable for faster startups

---

## Step 3: Install llcuda

```python
!pip install -q git+https://github.com/llcuda/llcuda.git@v2.2.0
```

---

## Step 4: Verify Setup

```python
import llcuda
from llcuda.api.multigpu import detect_gpus, print_gpu_info

print(f"llcuda v{llcuda.__version__}")
print_gpu_info()
```

---

## Step 5: Test Inference

Run the [Quick Start notebook](../tutorials/01-quickstart.md) to verify everything works.

---

## Kaggle Limits

- **Session Duration**: 12 hours maximum
- **Disk Space**: 73 GB available
- **VRAM**: 30 GB total (2× 15GB T4)
- **Internet**: Required for pip installs

---

## Next Steps

- [Multi-GPU Guide](../kaggle/multi-gpu-inference.md)
- [Tutorial Notebooks](../tutorials/index.md)
""")

print("\n✅ Documentation generation complete!")
print(f"\nCreated {len(created_files)} files:")
for f in created_files:
    print(f"  - {f}")

print("\n📝 Next steps:")
print("1. Review generated files")
print("2. Run: mkdocs serve")
print("3. View at: http://127.0.0.1:8000")
print("4. Deploy: mkdocs gh-deploy")
