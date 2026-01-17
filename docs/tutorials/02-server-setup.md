# Server Setup

Deep dive into llama-server configuration and lifecycle management.

**Level**: Beginner | **Time**: 15 minutes | **VRAM Required**: 5-8 GB (single T4)

---

## ServerConfig Parameters

```python
from llcuda.server import ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    context_size=4096,
    n_batch=2048,
    flash_attn=True,
    tensor_split=None,  # Single GPU
    host="127.0.0.1",
    port=8080
)
```

## Server Lifecycle

```python
from llcuda.server import ServerManager

server = ServerManager()

# Start
server.start_with_config(config)

# Check status
print(f"Running: {server.is_running()}")
print(f"URL: {server.get_base_url()}")

# Wait for ready
server.wait_until_ready(timeout=30)

# Get logs
logs = server.get_logs()

# Stop
server.stop()
```

## Multi-GPU Configuration

```python
config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",  # 50/50 split
    split_mode="layer",
    n_gpu_layers=99,
    flash_attn=True
)
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/02-llama-server-setup-llcuda-v2-2-0)
