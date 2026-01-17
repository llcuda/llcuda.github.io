# Troubleshooting Guide

Solutions to common issues with llcuda v2.2.0 on Tesla T4 GPUs.

## Installation Issues

### pip install fails

**Symptom:**
```bash
ERROR: Could not find a version that satisfies the requirement llcuda
```

**Solution:**
```bash
# Install from GitHub (not PyPI for v2.2.0)
pip install git+https://github.com/llcuda/llcuda.git

# Or use specific release
pip install https://github.com/llcuda/llcuda/releases/download/v2.2.0/llcuda-2.2.0-py3-none-any.whl
```

### Binary download fails

**Symptom:**
```
Failed to download CUDA binaries: HTTP 404
```

**Solution:**
```python
# Manually download binaries
import requests
import tarfile
from pathlib import Path

url = "https://github.com/llcuda/llcuda/releases/download/v2.2.0/llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz"
cache_dir = Path.home() / ".cache" / "llcuda"
cache_dir.mkdir(parents=True, exist_ok=True)

# Download
response = requests.get(url)
tar_path = cache_dir / "binaries.tar.gz"
tar_path.write_bytes(response.content)

# Extract
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(cache_dir)
```

## GPU Issues

### GPU not detected

**Symptom:**
```
CUDA not available
No CUDA GPU detected
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# If fails in Kaggle, verify accelerator type
# Settings > Accelerator > GPU T4 x 2

# Verify CUDA version
nvcc --version  # Should show CUDA 12.x
```

### Wrong GPU detected

**Symptom:**
```
Your GPU is not Tesla T4
GPU: Tesla P100 (SM 6.0)
```

**Solution:**
llcuda v2.2.0 is optimized for Kaggle dual Tesla T4. For other GPUs, compatibility may vary.

## Model Loading Issues

### Model not found

**Symptom:**
```
FileNotFoundError: Model file not found: gemma-3-1b-Q4_K_M
```

**Solution:**
```python
# Use full HuggingFace path
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
)

# Or download manually
from llcuda.models import download_model
model_path = download_model(
    "unsloth/gemma-3-1b-it-GGUF",
    "gemma-3-1b-it-Q4_K_M.gguf"
)
```

### Out of memory

**Symptom:**
```
CUDA out of memory
Failed to allocate tensor
```

**Solution:**
```python
# Reduce GPU layers
engine.load_model("model.gguf", gpu_layers=20)

# Reduce context size
engine.load_model("model.gguf", ctx_size=1024)

# Use smaller quantization
# Q4_K_M instead of Q8_0
```

## Server Issues

### Server won't start

**Symptom:**
```
RuntimeError: Failed to start llama-server
```

**Solution:**
```python
# Check if port is in use
import socket
sock = socket.socket()
try:
    sock.bind(('127.0.0.1', 8090))
    print("Port 8090 is free")
except:
    print("Port 8090 is in use - trying different port")
sock.close()

# Use different port
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8091")
```

### Server crashes

**Symptom:**
```
llama-server process died unexpectedly
```

**Solution:**
```python
# Run without silent mode to see errors
engine.load_model("model.gguf", silent=False, verbose=True)

# Try reducing memory usage
engine.load_model(
    "model.gguf",
    gpu_layers=20,
    ctx_size=1024
)
```

## Performance Issues

### Slow inference (<50 tok/s)

**Solutions:**
```python
# 1. Increase GPU offload
engine.load_model("model.gguf", gpu_layers=99)

# 2. Use Q4_K_M quantization
engine.load_model("model-Q4_K_M.gguf")

# 3. Reduce context
engine.load_model("model.gguf", ctx_size=2048)

# 4. Check GPU usage
!nvidia-smi  # Should show 80%+ GPU utilization
```

### High latency (>2000ms)

**Solution:**
```python
# Reduce max_tokens
result = engine.infer("Prompt", max_tokens=50)

# Use smaller model (Gemma 3-1B instead of Llama 3.1-8B)

# Optimize parameters
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=99,
    ctx_size=1024,
    batch_size=512
)
```

## Common Error Messages

### "Binaries not found"

```bash
# Reinstall with cache clear
pip uninstall llcuda -y
pip cache purge
pip install git+https://github.com/llcuda/llcuda.git --no-cache-dir
```

### "LD_LIBRARY_PATH not set"

```python
import os
from pathlib import Path

# Manually set library path
lib_dir = Path.home() / ".cache" / "llcuda" / "lib"
os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
```

### "CUDA version mismatch"

```bash
# Check CUDA version
nvcc --version
nvidia-smi  # Look for "CUDA Version"

# llcuda requires CUDA 12.0+
# Kaggle has CUDA 12.2+ by default
```

## Kaggle Specific

### T4 GPUs not available

**Solution:**
- In Kaggle: Settings > Accelerator > GPU T4 x 2
- Enable Internet access: Settings > Internet > On
- Dual T4 GPUs are always available on Kaggle (free tier)

### Session disconnects after 12 hours

**Solution:**
Kaggle has a 12-hour maximum session limit. Save your work to `/kaggle/working` which persists between sessions.

## Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", verbose=True, silent=False)
```

## Getting Help

1. **Check error details:**
   ```python
   result = engine.infer("test", max_tokens=10)
   if not result.success:
       print(f"Error: {result.error_message}")
   ```

2. **GitHub Issues:** [github.com/llcuda/llcuda/issues](https://github.com/llcuda/llcuda/issues)

3. **Include in bug reports:**
   - llcuda version (`llcuda.__version__`)
   - GPU model (`nvidia-smi`)
   - CUDA version (`nvcc --version`)
   - Python version (`python --version`)
   - Full error message
   - Minimal reproducible code

## Quick Fixes Checklist

- [ ] GPU is Tesla T4 (check with `nvidia-smi`)
- [ ] CUDA 12.0+ installed (check with `nvcc --version`)
- [ ] Latest llcuda from GitHub (`pip install git+https://github.com/llcuda/llcuda.git`)
- [ ] Model exists and is accessible
- [ ] Port 8090 is available
- [ ] Sufficient VRAM for model
- [ ] Using Q4_K_M quantization
- [ ] gpu_layers=99 for full offload

## Next Steps

- [FAQ](faq.md) - Frequently asked questions
- [Performance Optimization](../tutorials/performance.md) - Speed up inference
- [First Steps](first-steps.md) - Getting started guide
- [GitHub Issues](https://github.com/llcuda/llcuda/issues) - Report bugs
