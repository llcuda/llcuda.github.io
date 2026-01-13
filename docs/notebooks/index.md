# Google Colab Notebooks

Complete collection of ready-to-run Jupyter notebooks for llcuda v2.1.0 on Google Colab with Tesla T4 GPU.

## Overview

llcuda includes 8 comprehensive Google Colab notebooks covering installation, inference, fine-tuning workflows, and binary building. All notebooks are optimized for Tesla T4 GPUs and include detailed explanations, code examples, and performance metrics.

## Available Notebooks

### 1. Gemma 3-1B Tutorial (Recommended)

**File:** `llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb`

Complete guide for using llcuda v2.1.0 with Unsloth GGUF models on Tesla T4 GPU.

**What it covers:**
- ✅ Install llcuda v2.1.0 from GitHub
- ✅ Auto-download CUDA binaries from GitHub Releases
- ✅ Load Gemma 3-1B-IT GGUF from Unsloth
- ✅ Fast inference with FlashAttention (134 tok/s verified)
- ✅ Batch processing and performance metrics
- ✅ Advanced generation parameters
- ✅ Unsloth fine-tuning → llcuda deployment workflow

**Time required:** ~10 minutes

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

[:material-file-document: View Tutorial](../tutorials/gemma-3-1b-colab.md){ .md-button }

---

### 2. Gemma 3-1B Executed Example

**File:** `llcuda_v2_1_0_gemma3_1b_unsloth_colab_executed.ipynb`

Live execution output from Tesla T4 GPU showing real performance results.

**What it shows:**
- ✅ Complete output from all cells
- ✅ **Verified 134 tok/s performance** on Gemma 3-1B Q4_K_M
- ✅ Real GPU metrics and timings
- ✅ Proof of working binary download and model loading
- ✅ Batch inference results (130-142 tok/s range)

**Why it's useful:**
- See exactly what to expect on Tesla T4
- Verify performance before running
- Understand output format
- Debugging reference

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab_executed.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

[:material-chart-line: View Executed Results](../tutorials/gemma-3-1b-executed.md){ .md-button }

---

### 3. Build llcuda Binaries on T4

**File:** `build_llcuda_v2_t4_colab.ipynb`

Build CUDA 12 binaries from source on Tesla T4 GPU.

**What it covers:**
- ✅ Clone and build llama.cpp with CUDA 12
- ✅ Enable FlashAttention and Tensor Core optimization
- ✅ Compile with SM 7.5 targeting (Tesla T4)
- ✅ Create binary packages for release
- ✅ Download complete package (~350-400 MB)

**Time required:** ~15-20 minutes

**When to use:**
- Building from source
- Creating custom binary packages
- Contributing to llcuda development
- Understanding the build process

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/build_llcuda_v2_t4_colab.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

[:material-hammer-wrench: View Build Guide](../tutorials/build-binaries.md){ .md-button }

---

### 4. Unsloth + llcuda Complete Build

**File:** `llcuda_unsloth_t4_complete_build.ipynb`

Complete build workflow combining llama.cpp and llcuda for Tesla T4.

**What it covers:**
- ✅ Build llama.cpp with FlashAttention
- ✅ Build llcuda Python package
- ✅ Create unified tar file with everything
- ✅ One-package distribution

**Output:** `llcuda-complete-cuda12-t4.tar.gz` (~350-400 MB)

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_unsloth_t4_complete_build.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

---

### 5. Unsloth Tutorial

**File:** `llcuda_unsloth_tutorial.ipynb`

Usage guide demonstrating llcuda with Unsloth GGUF models.

**What it covers:**
- ✅ Install llcuda (auto-downloads binaries)
- ✅ Load Unsloth GGUF models
- ✅ Fast inference demonstrations
- ✅ Batch processing examples
- ✅ Unsloth → llcuda workflow

**Time required:** ~5-10 minutes

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_unsloth_tutorial.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

---

### 6. llcuda Quickstart Tutorial

**File:** `llcuda_quickstart_tutorial.ipynb`

Quick introduction to llcuda basics.

**What it covers:**
- ✅ Basic installation
- ✅ Simple inference examples
- ✅ Model loading methods
- ✅ Performance metrics

**Time required:** ~5 minutes

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_quickstart_tutorial.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

---

### 7. Advanced Example: p3_llcuda

**File:** `p3_llcuda.ipynb`

Advanced usage patterns and optimization techniques.

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/p3_llcuda.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

---

### 8. Advanced Example: p3_1_llcuda

**File:** `p3_1_llcuda.ipynb`

Extended advanced examples with additional features.

**Open in Colab:**
<div style="text-align: center; margin: 1em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/p3_1_llcuda.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</div>

---

## How to Use These Notebooks

### Running on Google Colab

1. **Click "Open in Colab" button** on any notebook above
2. **Set runtime to T4 GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (if available)
   - Click **Save**
3. **Run all cells:**
   - Runtime → Run all
   - Or press Shift+Enter on each cell
4. **Wait for completion** (time varies by notebook)

### Saving Your Work

```python
# Save results to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy outputs
!cp output.txt /content/drive/MyDrive/llcuda_results/
```

### Downloading Generated Files

```python
from google.colab import files

# Download any generated file
files.download('model_output.txt')
```

## Notebook Categories

### For Beginners
- ✅ **Gemma 3-1B Tutorial** - Start here!
- ✅ **Quickstart Tutorial** - 5-minute introduction
- ✅ **Unsloth Tutorial** - Unsloth integration

### For Advanced Users
- ✅ **Build Binaries** - Compile from source
- ✅ **Complete Build** - Full build workflow
- ✅ **p3/p3_1 Examples** - Advanced patterns

### For Verification
- ✅ **Gemma 3-1B Executed** - See real T4 results

## Common Issues

### Issue: Runtime Disconnected

**Solution:**
- Keep Colab tab active
- Use Colab Pro for longer runtimes
- Save checkpoints regularly

### Issue: GPU Not Available

**Solution:**
```python
# Check GPU status
!nvidia-smi

# If no GPU, change runtime:
# Runtime → Change runtime type → GPU (T4)
```

### Issue: Out of Memory

**Solution:**
- Use smaller models (Gemma 3-1B instead of 8B)
- Clear runtime: Runtime → Restart runtime
- Use lower quantization (Q4_K_M recommended)

## Performance Expectations

| Notebook | Download Size | Runtime | Expected Speed |
|----------|--------------|---------|----------------|
| Gemma 3-1B Tutorial | ~916 MB | ~10 min | 134 tok/s |
| Build Binaries | ~2 GB | ~20 min | Build only |
| Quickstart | ~650 MB | ~5 min | Variable |
| Unsloth Tutorial | ~650 MB | ~10 min | ~45 tok/s |

## Next Steps

After running the notebooks:

- [:material-api: API Reference](../api/overview.md) - Detailed API documentation
- [:material-tune: Performance Optimization](../performance/optimization.md) - Get better performance
- [:material-book-open: Unsloth Integration](../tutorials/unsloth-integration.md) - Complete workflow
- [:material-frequently-asked-questions: FAQ](../guides/faq.md) - Common questions

---

**All notebooks are maintained at:** [github.com/waqasm86/llcuda/tree/main/notebooks](https://github.com/waqasm86/llcuda/tree/main/notebooks)
