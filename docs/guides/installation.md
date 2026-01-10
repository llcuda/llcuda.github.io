# Installation Guide

Install llcuda v2.0.6 directly from GitHub - **No PyPI needed!**

## :rocket: Quick Install

### Method 1: Direct from GitHub (Recommended)

```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

This single command will:

- ✅ Clone the latest code from GitHub
- ✅ Install the Python package
- ✅ Auto-download CUDA binaries (266 MB) from GitHub Releases on first import

!!! success "Recommended for most users"
    This is the easiest method and works perfectly on Google Colab, Kaggle, and local systems.

### Method 2: Install from Specific Release

```bash
pip install https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-2.0.6-py3-none-any.whl
```

### Method 3: Install from Source (Development)

```bash
git clone https://github.com/waqasm86/llcuda.git
cd llcuda
pip install -e .
```

---

## :package: What Gets Installed

### Python Package
- **Source:** GitHub repository (main branch or release tag)
- **Size:** ~100 KB (Python code only, no binaries)
- **Contents:** Core Python package, API, bootstrap code

### CUDA Binaries (Auto-Downloaded)
- **Source:** [GitHub Releases v2.0.6](https://github.com/waqasm86/llcuda/releases/tag/v2.0.6)
- **URL:** `llcuda-binaries-cuda12-t4-v2.0.6.tar.gz`
- **Size:** 266 MB (one-time download, cached locally)
- **Triggered:** On first `import llcuda`
- **Location:** `~/.cache/llcuda/` or `<package>/binaries/`

**Binary Package Contents:**
```
llcuda-binaries-cuda12-t4-v2.0.6.tar.gz (266 MB)
├── bin/
│   ├── llama-server        (6.5 MB) - Inference server
│   ├── llama-cli           (4.2 MB) - Command-line interface
│   ├── llama-embedding     (3.3 MB) - Embedding generator
│   ├── llama-bench         (581 KB) - Benchmarking tool
│   └── llama-quantize      (434 KB) - Model quantization
└── lib/
    ├── libggml-cuda.so     (221 MB) - CUDA kernels + FlashAttention
    ├── libllama.so         (2.9 MB) - Llama core library
    └── Other libraries...
```

---

## :gear: Platform-Specific Instructions

=== "Google Colab"
    ### Google Colab (Tesla T4)

    Perfect for cloud notebooks!

    ```python
    # 1. Install
    !pip install -q git+https://github.com/waqasm86/llcuda.git

    # 2. Import (triggers binary download on first run)
    import llcuda

    # 3. Verify GPU
    compat = llcuda.check_gpu_compatibility()
    print(f"GPU: {compat['gpu_name']}")        # Should show: Tesla T4
    print(f"Compatible: {compat['compatible']}") # Should show: True

    # 4. Ready to use!
    engine = llcuda.InferenceEngine()
    ```

    !!! tip "First Run"
        The first import downloads 266 MB of binaries (takes 1-2 minutes).
        Subsequent sessions reuse cached binaries - instant startup!

=== "Local Linux"
    ### Local Linux (Ubuntu/Debian)

    **Requirements:**
    - Python 3.11+
    - CUDA 12.x runtime
    - Tesla T4 GPU or compatible

    ```bash
    # 1. Ensure CUDA 12 is installed
    nvidia-smi  # Should show CUDA 12.x

    # 2. Install llcuda
    pip install git+https://github.com/waqasm86/llcuda.git

    # 3. Test installation
    python3 -c "import llcuda; print(llcuda.__version__)"
    # Output: 2.0.6
    ```

    **System Dependencies (usually pre-installed):**
    ```bash
    # CUDA Runtime (required)
    sudo apt install nvidia-cuda-toolkit

    # Python dependencies (installed automatically by pip)
    # - requests
    # - numpy
    ```

=== "Kaggle"
    ### Kaggle Notebooks

    ```python
    # 1. Enable GPU accelerator
    # Settings → Accelerator → GPU T4 x2

    # 2. Install
    !pip install -q git+https://github.com/waqasm86/llcuda.git

    # 3. Import and verify
    import llcuda
    compat = llcuda.check_gpu_compatibility()
    print(f"GPU: {compat['gpu_name']}")

    # 4. Start using
    engine = llcuda.InferenceEngine()
    ```

=== "Windows (WSL2)"
    ### Windows with WSL2

    **Prerequisites:**
    - Windows 11 with WSL2
    - NVIDIA GPU with CUDA support
    - CUDA 12.x installed in WSL2

    ```bash
    # Inside WSL2 terminal
    # 1. Verify CUDA
    nvidia-smi

    # 2. Install Python 3.11+
    sudo apt install python3.11 python3-pip

    # 3. Install llcuda
    pip3 install git+https://github.com/waqasm86/llcuda.git

    # 4. Test
    python3 -c "import llcuda; print(llcuda.__version__)"
    ```

---

## :white_check_mark: Verification

After installation, verify everything works:

```python
import llcuda

# 1. Check version
print(f"llcuda version: {llcuda.__version__}")
# Expected: 2.0.6

# 2. Check GPU compatibility
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute Capability: SM {compat['compute_capability']}")
print(f"Platform: {compat['platform']}")
print(f"Compatible: {compat['compatible']}")

# 3. Quick inference test
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

result = engine.infer("What is 2+2?", max_tokens=20)
print(f"Response: {result.text}")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
# Expected on T4: ~134 tok/s
```

---

## :wrench: Manual Binary Installation (Advanced)

If automatic download fails, install binaries manually:

```bash
# 1. Download binary package
wget https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz

# 2. Verify checksum
echo "5a27d2e1a73ae3d2f1d2ba8cf557b76f54200208c8df269b1bd0d9ee176bb49d  llcuda-binaries-cuda12-t4-v2.0.6.tar.gz" | sha256sum -c

# 3. Extract to cache directory
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-binaries-cuda12-t4-v2.0.6.tar.gz -C ~/.cache/llcuda/

# 4. Or extract to package directory
python3 -c "import llcuda; print(llcuda._BIN_DIR)"
# Extract to the printed directory
```

---

## :test_tube: Testing Your Installation

### Basic Test

```python
import llcuda

# Should not raise any errors
print("✅ llcuda imported successfully")
```

### GPU Test

```python
compat = llcuda.check_gpu_compatibility()
assert compat['compatible'], "GPU not compatible!"
print(f"✅ GPU compatible: {compat['gpu_name']}")
```

### Inference Test

```python
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

result = engine.infer("Hello!", max_tokens=10)
assert result.tokens_generated > 0
print(f"✅ Inference working: {result.tokens_per_sec:.1f} tok/s")
```

---

## :construction: Requirements

### System Requirements

- **Python:** 3.11 or higher
- **CUDA:** 12.x runtime
- **GPU:** Tesla T4 (SM 7.5) - **Primary target**
- **RAM:** 4 GB minimum
- **Disk:** 1 GB free space (for binaries and models)

### Python Dependencies

Automatically installed by pip:

```
requests>=2.31.0
numpy>=1.24.0
```

Optional for development:

```
huggingface-hub>=0.19.0  # For model downloads
```

---

## :arrows_counterclockwise: Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade git+https://github.com/waqasm86/llcuda.git
```

### Force Reinstall

```bash
pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/waqasm86/llcuda.git
```

### Clear Cache and Reinstall

```bash
# Remove cached binaries
rm -rf ~/.cache/llcuda/

# Reinstall
pip uninstall llcuda -y
pip install git+https://github.com/waqasm86/llcuda.git
```

---

## :x: Uninstallation

```bash
# Remove Python package
pip uninstall llcuda -y

# Remove cached binaries
rm -rf ~/.cache/llcuda/

# Remove package installation
pip cache purge
```

---

## :question: Troubleshooting

### Binary Download Fails

**Error:** `RuntimeError: Binary download failed`

**Solution:**
```python
# Check internet connection
import requests
response = requests.get("https://github.com")
print(response.status_code)  # Should be 200

# Try manual download (see Manual Binary Installation above)
```

### Import Error

**Error:** `ModuleNotFoundError: No module named 'llcuda'`

**Solution:**
```bash
# Verify installation
pip list | grep llcuda

# Reinstall
pip install --force-reinstall git+https://github.com/waqasm86/llcuda.git
```

### GPU Not Detected

**Error:** `RuntimeError: No CUDA GPUs detected`

**Solution:**
```bash
# Verify CUDA is working
nvidia-smi

# Check GPU visibility
python3 -c "import subprocess; print(subprocess.run(['nvidia-smi'], capture_output=True).stdout)"
```

---

## :link: Next Steps

- [:material-rocket-launch: Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [:material-google: Google Colab Tutorial](../tutorials/gemma-3-1b-colab.md) - Complete walkthrough
- [:material-help-circle: Troubleshooting](troubleshooting.md) - Common issues and solutions
- [:material-api: API Reference](../api/overview.md) - Detailed API documentation

---

**Need help?** [Open an issue on GitHub](https://github.com/waqasm86/llcuda/issues)
