# LlamaCppClient API

OpenAI-compatible client for llama-server.

## Overview

The `LlamaCppClient` provides an OpenAI-compatible interface to llama-server.

## Basic Usage

```python
from llcuda.api.client import LlamaCppClient

client = LlamaCppClient(base_url="http://localhost:8080")

response = client.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

## Class Reference

### `LlamaCppClient`

```python
class LlamaCppClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize client.
        
        Args:
            base_url: Base URL of llama-server
        """
```

### Methods

#### `create_chat_completion()`

```python
def create_chat_completion(
    self,
    messages: List[Dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False
) -> Dict:
    """Create chat completion.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stream: Enable streaming
        
    Returns:
        Response dict with 'choices', 'usage', etc.
    """
```

## Examples

See [API Examples](examples.md)
