# LlamaCppClient API

OpenAI-compatible and native llama.cpp client for llama-server.

## Overview

The `LlamaCppClient` provides OpenAI-compatible chat/completions plus native endpoints (complete, embeddings, tokenize).

## Basic Usage

```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient(base_url="http://localhost:8080")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
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

#### `chat.completions.create()`

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)
```

#### `complete()`

```python
response = client.complete(
    prompt="The future of AI is",
    n_predict=50,
    temperature=0.8
)
```

#### `embeddings.create()`

```python
emb = client.embeddings.create(input=["Hello world", "Goodbye world"])
```

#### `tokenize()` / `detokenize()`

```python
tokens = client.tokenize("Hello, world!")
text = client.detokenize(tokens.tokens)
```

## Examples

See [API Examples](examples.md) and [InferenceEngine](inference-engine.md) for a simpler wrapper.
