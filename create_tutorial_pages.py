#!/usr/bin/env python3.11
"""Generate missing tutorial pages for llcuda v2.2.0 Kaggle notebooks."""

import os
from pathlib import Path

# Base directory
DOCS_DIR = Path(__file__).parent / "docs"
TUTORIALS_DIR = DOCS_DIR / "tutorials"
TUTORIALS_DIR.mkdir(parents=True, exist_ok=True)

# Tutorial definitions based on llcuda v2.2.0 notebooks
TUTORIALS = [
    {
        "num": "01",
        "slug": "quickstart",
        "title": "Quick Start",
        "description": "Get started with llcuda in 5 minutes",
        "time": "5 min",
        "level": "Beginner",
        "prereq": "None",
        "vram": "3-5 GB (single T4)",
        "content": """Get started with llcuda v2.2.0 in just 5 minutes on Kaggle dual T4 GPUs.

## Overview

This tutorial covers the essentials:

- Installing llcuda v2.2.0
- Downloading a GGUF model
- Starting the llama-server
- Making your first chat completion
- Cleaning up resources

## Prerequisites

- **Kaggle account** with GPU quota
- **Accelerator**: GPU T4 × 2
- **Internet**: Enabled for package installation

## Step 1: Install llcuda

```bash
pip install llcuda
```

On first import, llcuda will auto-download the 961 MB binary package containing llama.cpp build 7760 with FlashAttention.

## Step 2: Import and Initialize

```python
from llcuda.server import ServerManager, ServerConfig

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
```

## Step 3: Download a Model

```python
# Download Gemma 2-2B Q4_K_M from HuggingFace
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-2-2b-it-GGUF",
    filename="gemma-2-2b-it-Q4_K_M.gguf"
)
print(f"Model downloaded to: {model_path}")
```

## Step 4: Start the Server

```python
# Configure for single GPU
config = ServerConfig(
    model_path=model_path,
    n_gpu_layers=99,  # Offload all layers to GPU
    context_size=4096,
    flash_attn=True
)

# Start server
server = ServerManager()
server.start_with_config(config)
print("Server started successfully!")
```

## Step 5: Make Your First Request

```python
from llcuda.api.client import LlamaCppClient

# Create client
client = LlamaCppClient(base_url="http://localhost:8080")

# Chat completion
response = client.create_chat_completion(
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    max_tokens=200
)

print(response["choices"][0]["message"]["content"])
print(f"Tokens/sec: {response['usage']['tokens_per_sec']:.1f}")
```

## Step 6: Cleanup

```python
# Stop server
server.stop()
print("Server stopped")
```

## Expected Performance

On Kaggle dual T4 with Gemma 2-2B Q4_K_M:

- **Speed**: ~60 tokens/sec
- **Latency**: ~500ms
- **VRAM**: ~3-4 GB

## Next Steps

- [02 - Server Setup](02-server-setup.md) - Deep dive into server configuration
- [03 - Multi-GPU Inference](03-multi-gpu.md) - Use both T4 GPUs
- [API Reference](../api/server.md) - Complete ServerManager API

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/01-quickstart-llcuda-v2-2-0)
"""
    },
    {
        "num": "02",
        "slug": "server-setup",
        "title": "Server Setup",
        "description": "Deep dive into llama-server configuration",
        "time": "15 min",
        "level": "Beginner",
        "prereq": "Complete notebook 01",
        "vram": "5-8 GB (single T4)",
        "content": """Complete guide to configuring and managing the llama-server lifecycle.

## Overview

Learn about:

- ServerConfig parameter reference
- Server lifecycle (start → ready → stop)
- Health checking and monitoring
- Log access and debugging
- Multiple server configurations

## ServerConfig Parameters

```python
from llcuda.server import ServerConfig

config = ServerConfig(
    model_path="model.gguf",         # Required: Path to GGUF model
    n_gpu_layers=99,                 # GPU layers (99 = all)
    context_size=4096,               # Context window
    n_batch=2048,                    # Batch size
    flash_attn=True,                 # Enable FlashAttention
    tensor_split=None,               # Single GPU (default)
    host="127.0.0.1",               # Server host
    port=8080,                       # Server port
)
```

## Server Lifecycle

```python
from llcuda.server import ServerManager

server = ServerManager()

# Start server
server.start_with_config(config)

# Check if running
if server.is_running():
    print("Server is running")

# Get server URL
print(f"Server URL: {server.get_base_url()}")

# Wait for ready
server.wait_until_ready(timeout=30)

# Get logs
logs = server.get_logs()
print(logs)

# Stop server
server.stop()
```

## Health Checking

```python
import requests

# Check server health
response = requests.get("http://localhost:8080/health")
print(response.json())

# Output:
# {"status": "ok", "slots_idle": 1, "slots_processing": 0}
```

## Advanced Configuration

### Multi-GPU Setup

```python
config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",  # Split 50/50 across 2 GPUs
    split_mode="layer",      # Layer-wise splitting
    n_gpu_layers=99,
    flash_attn=True
)
```

### Memory Optimization

```python
config = ServerConfig(
    model_path="model.gguf",
    context_size=2048,       # Smaller context
    n_batch=512,             # Smaller batch
    n_gpu_layers=99,
    flash_attn=True
)
```

## Debugging

```python
# Enable verbose logging
server.start_with_config(config, verbose=True)

# Access logs
logs = server.get_logs()
for line in logs.split('\\n')[-20:]:  # Last 20 lines
    print(line)
```

## Next Steps

- [03 - Multi-GPU Inference](03-multi-gpu.md) - Use dual T4 GPUs
- [API Reference](../api/server.md) - Complete ServerManager API
- [Troubleshooting](../guides/troubleshooting.md) - Common issues

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/02-llama-server-setup-llcuda-v2-2-0)
"""
    },
]

def create_tutorial(tutorial):
    """Create a single tutorial markdown file."""
    filename = f"{tutorial['num']}-{tutorial['slug']}.md"
    filepath = TUTORIALS_DIR / filename

    # Create frontmatter and content
    content = f"""# {tutorial['title']}

{tutorial['description']}

**Level**: {tutorial['level']}
**Time**: {tutorial['time']}
**Prerequisites**: {tutorial['prereq']}
**VRAM Required**: {tutorial['vram']}

---

{tutorial['content']}

---

**Questions?** [Open an issue on GitHub](https://github.com/llcuda/llcuda/issues)
"""

    filepath.write_text(content)
    print(f"✓ Created {filename}")

def main():
    """Generate all tutorial pages."""
    print("Generating llcuda v2.2.0 tutorial pages...")
    print()

    for tutorial in TUTORIALS:
        create_tutorial(tutorial)

    print()
    print(f"✓ Generated {len(TUTORIALS)} tutorial pages in {TUTORIALS_DIR}")

if __name__ == "__main__":
    main()
