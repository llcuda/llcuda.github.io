# Google Colab Usage Guide

Complete guide for using llcuda v2.0.6 on Google Colab with Tesla T4 GPUs.

## Why Google Colab?

Google Colab provides **free Tesla T4 GPU access**, making it perfect for running llcuda:

✅ **Free Tesla T4 GPU** (up to 12 hours per session)
✅ **No local setup required** (runs in browser)
✅ **Pre-installed CUDA 12.x** (ready for llcuda)
✅ **Python 3.10+ environment** (compatible)
✅ **Easy sharing** (share notebooks via links)

## Quick Start on Colab

### Step 1: Open a Notebook

Click any "Open in Colab" button from the [Notebooks Index](index.md), or create a new notebook:

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **New Notebook**
3. File → Save (to your Google Drive)

### Step 2: Enable T4 GPU

!!! warning "Critical Step"
    You **must** enable T4 GPU runtime for llcuda to work!

**Steps:**
1. Click **Runtime** in the menu
2. Select **Change runtime type**
3. Set **Hardware accelerator** to **GPU**
4. Set **GPU type** to **T4** (if option available in free tier)
5. Click **Save**

**Verify GPU is active:**

```python
# Check GPU
!nvidia-smi

# Should show: Tesla T4
# CUDA Version: 12.x
```

### Step 3: Install llcuda

```python
# Install from GitHub
!pip install -q git+https://github.com/waqasm86/llcuda.git

# Import (triggers binary download on first run)
import llcuda

# Verify installation
print(f"llcuda version: {llcuda.__version__}")

# Check GPU compatibility
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compatible: {compat['compatible']}")
```

**Expected output:**
```
llcuda version: 2.0.6
GPU: Tesla T4
Compatible: True
```

### Step 4: Run Inference

```python
# Initialize engine
engine = llcuda.InferenceEngine()

# Load model from Unsloth
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Run inference
result = engine.infer(
    "Explain quantum computing in simple terms",
    max_tokens=200
)

print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

## Colab Features for llcuda

### 1. Persistent Storage with Google Drive

Mount Google Drive to save models and outputs:

```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Save outputs to Drive
output_dir = '/content/drive/MyDrive/llcuda_outputs/'
!mkdir -p {output_dir}

# Save inference results
with open(f'{output_dir}/results.txt', 'w') as f:
    f.write(result.text)
```

### 2. Download Files

Download generated files to your computer:

```python
from google.colab import files

# Generate and download results
with open('inference_results.txt', 'w') as f:
    f.write(result.text)

files.download('inference_results.txt')
```

### 3. Upload Files

Upload local files to Colab:

```python
from google.colab import files

# Upload a GGUF model file
uploaded = files.upload()

# Use uploaded file
for filename in uploaded.keys():
    print(f"Uploaded: {filename}")
    engine.load_model(filename)
```

### 4. Display Rich Output

```python
from IPython.display import Markdown, display

# Display formatted output
display(Markdown(f"""
## Inference Results

**Prompt:** {prompt}

**Response:**
{result.text}

**Performance:**
- Speed: {result.tokens_per_sec:.1f} tok/s
- Latency: {result.latency_ms:.1f} ms
- Tokens: {result.tokens_generated}
"""))
```

### 5. Progress Bars

Show progress for batch processing:

```python
from tqdm import tqdm

prompts = ["prompt 1", "prompt 2", "prompt 3"]
results = []

for prompt in tqdm(prompts, desc="Processing"):
    result = engine.infer(prompt, max_tokens=100)
    results.append(result)
```

## Runtime Management

### Check Runtime Status

```python
# Check GPU memory
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check RAM usage
!free -h

# Check disk space
!df -h
```

### Restart Runtime

If you encounter issues:

1. **Runtime** → **Restart runtime**
2. Re-run installation cells
3. Models will need to be re-downloaded

### Extend Session Time

**Colab Free:**
- 12 hours max per session
- Keep tab active to avoid disconnection
- Use `%%capture` to suppress verbose output

**Colab Pro:**
- 24 hours max per session
- Background execution available
- Priority access to T4 GPUs

## Best Practices for Colab

### 1. Cache Models Efficiently

```python
import os
from pathlib import Path

# Set cache directory
cache_dir = Path.home() / ".cache" / "llcuda"
os.environ['LLCUDA_CACHE_DIR'] = str(cache_dir)

# Models are cached here (persist across cells)
print(f"Cache: {cache_dir}")
```

### 2. Silent Mode for Servers

Suppress llama-server output:

```python
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True  # ← Suppress server logs
)
```

### 3. Cleanup Resources

```python
# Stop inference engine
engine.stop()

# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Check freed memory
!nvidia-smi
```

### 4. Use Context Managers

Automatic cleanup when done:

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("model.gguf", silent=True)
    result = engine.infer("prompt", max_tokens=100)
    print(result.text)
# Engine automatically stopped here
```

## Optimizing for Colab

### 1. Model Selection

Choose models that fit in T4's 16 GB VRAM:

| Model | Quantization | VRAM | Speed | Fits T4? |
|-------|--------------|------|-------|----------|
| Gemma 3-1B | Q4_K_M | 1.2 GB | 134 tok/s | ✅ Perfect |
| Llama 3.2-3B | Q4_K_M | 2.0 GB | ~30 tok/s | ✅ Yes |
| Qwen 2.5-7B | Q4_K_M | 5.0 GB | ~18 tok/s | ✅ Yes |
| Llama 3.1-8B | Q4_K_M | 5.5 GB | ~15 tok/s | ✅ Yes |
| Llama 3.1-70B | Q4_K_M | 40 GB | N/A | ❌ Too large |

### 2. Batch Processing

Process multiple prompts efficiently:

```python
# Batch inference (faster than loop)
prompts = [
    "What is AI?",
    "Explain ML.",
    "Define DL."
]

results = engine.batch_infer(prompts, max_tokens=80)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

### 3. Reduce Downloads

**First run in session:**
```python
# Downloads binaries (~266 MB) + model (~650 MB)
# Total: ~916 MB, takes 2-3 minutes
```

**Subsequent runs in same session:**
```python
# Uses cached binaries and models
# Instant startup!
```

## Troubleshooting Colab Issues

### Issue: GPU Not Available

**Error:** `GPU not detected` or `CUDA not available`

**Solution:**
```python
# 1. Check runtime type
# Runtime → Change runtime type → GPU

# 2. Verify GPU
!nvidia-smi

# 3. If still no GPU, runtime might be out of quota
# Try again later or upgrade to Colab Pro
```

### Issue: Session Disconnected

**Error:** "Runtime disconnected"

**Solution:**
- Keep Colab tab active (minimize, don't close)
- Avoid long-running cells (>30 minutes)
- Use Colab Pro for longer sessions
- Save checkpoints to Drive regularly

### Issue: Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```python
# 1. Use smaller model
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
)  # Only 1.2 GB VRAM

# 2. Clear GPU cache
import torch
torch.cuda.empty_cache()

# 3. Restart runtime
# Runtime → Restart runtime
```

### Issue: Slow Downloads

**Error:** Downloads taking too long

**Solution:**
```python
# Use lighter quantization
# Q4_K_M (~650 MB) instead of Q8_0 (~1.2 GB)

# Or pre-download to Drive and load from there
engine.load_model('/content/drive/MyDrive/models/model.gguf')
```

### Issue: Binary Download Failed

**Error:** `Failed to download binaries`

**Solution:**
```bash
# Manual download
!wget https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.3.tar.gz
!mkdir -p ~/.cache/llcuda/
!tar -xzf llcuda-binaries-cuda12-t4-v2.0.3.tar.gz -C ~/.cache/llcuda/

# Retry import
import llcuda
print("Success!")
```

## Example Workflows

### Workflow 1: Quick Testing

```python
# Install and test in under 5 minutes
!pip install -q git+https://github.com/waqasm86/llcuda.git

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

result = engine.infer("Hello!", max_tokens=50)
print(result.text)
```

### Workflow 2: Batch Analysis

```python
# Analyze dataset with batch processing
import pandas as pd

# Sample data
df = pd.DataFrame({
    'text': ["Sample 1", "Sample 2", "Sample 3"]
})

# Process all rows
results = engine.batch_infer(
    df['text'].tolist(),
    max_tokens=100
)

# Save to Drive
df['summary'] = [r.text for r in results]
df.to_csv('/content/drive/MyDrive/results.csv')
```

### Workflow 3: Interactive Chat

```python
# Simple chat interface
print("Chat with Gemma 3-1B (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    result = engine.infer(user_input, max_tokens=300)
    print(f"Assistant: {result.text}\n")
```

## Colab Pro Benefits for llcuda

| Feature | Free | Pro | Pro+ |
|---------|------|-----|------|
| **Runtime** | 12 hours | 24 hours | 24 hours |
| **T4 Access** | Sometimes | Priority | Priority |
| **RAM** | 12 GB | 32 GB | 52 GB |
| **Background** | ❌ | ✅ | ✅ |
| **Cost** | Free | $10/mo | $50/mo |

**Recommendation:** Free tier is sufficient for most llcuda use cases!

## Sharing Your Notebooks

### Share Read-Only

1. File → Share
2. Copy link
3. Share with "Viewer" access

### Share Editable

1. File → Share
2. Set to "Editor" access
3. Recipients can run and modify

### Publish to GitHub

```python
# Save notebook to GitHub directly
# File → Save a copy in GitHub
# Select repository and path
```

## Next Steps

- [:material-notebook: View All Notebooks](index.md) - Browse available notebooks
- [:material-api: API Reference](../api/overview.md) - Detailed API docs
- [:material-tune: Performance Tips](../performance/optimization.md) - Optimize performance
- [:material-frequently-asked-questions: FAQ](../guides/faq.md) - Common questions

---

**Happy coding on Colab!** 🚀

For issues, visit [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
