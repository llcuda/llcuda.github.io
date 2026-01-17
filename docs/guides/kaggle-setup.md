# Kaggle Setup Guide

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
