# ServerManager API

Server lifecycle management for llama-server.

## Overview

`ServerManager` handles starting, stopping, and monitoring llama-server.

## Basic Usage

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(model_path="model.gguf", n_gpu_layers=99)
server = ServerManager()
server.start_with_config(config)
server.stop()
```

## Class Reference

### `ServerManager`

```python
class ServerManager:
    def start_with_config(self, config: ServerConfig, verbose: bool = False) -> None:
        """Start server with configuration."""
        
    def stop(self) -> None:
        """Stop the running server."""
        
    def is_running(self) -> bool:
        """Check if server is running."""
        
    def wait_until_ready(self, timeout: int = 30) -> bool:
        """Wait for server to be ready."""
        
    def get_base_url(self) -> str:
        """Get server URL."""
        
    def get_logs(self) -> str:
        """Get server logs."""
```

### `ServerConfig`

```python
class ServerConfig:
    model_path: str
    n_gpu_layers: int = 99
    context_size: int = 4096
    n_batch: int = 2048
    flash_attn: bool = True
    tensor_split: Optional[str] = None
    split_mode: str = "layer"
    host: str = "127.0.0.1"
    port: int = 8080
```

## Examples

See [Tutorials](../tutorials/index.md)
