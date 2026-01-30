# InferenceEngine

High-level interface for llama-server inference with automatic server management.

## Import

```python
import llcuda

engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8090")
```

## Basic Workflow

```python
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    auto_start=True,
)
result = engine.infer("What is llcuda?", max_tokens=128)
print(result.text)
```

## Methods

### `__init__(server_url: str = "http://127.0.0.1:8090")`
Create a new engine instance pointing at a llama-server.

### `check_server() -> bool`
Return `True` if the server health check succeeds.

### `load_model(model_name_or_path: str, gpu_layers: int | None = None, ctx_size: int | None = None, auto_start: bool = True, auto_configure: bool = True, n_parallel: int = 1, verbose: bool = True, interactive_download: bool = True, silent: bool = False, **kwargs) -> bool`
Load a GGUF model and (optionally) start llama-server automatically.

Supported `model_name_or_path` formats:
- Registry name: `gemma-3-1b-Q4_K_M`
- Local path: `/path/to/model.gguf`
- Hugging Face: `repo/name:filename.gguf`

### `infer(prompt: str, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40, seed: int = 0, stop_sequences: list[str] | None = None) -> InferResult`
Run single‑prompt inference and return an `InferResult`.

### `infer_stream(prompt: str, callback, max_tokens: int = 128, temperature: float = 0.7, **kwargs) -> InferResult`
Run inference and invoke `callback` with the generated text.

### `batch_infer(prompts: list[str], max_tokens: int = 128, **kwargs) -> list[InferResult]`
Run inference across multiple prompts.

### `get_metrics() -> dict`
Return aggregated latency/throughput metrics.

### `reset_metrics()`
Reset metrics counters.

### `unload_model()`
Stop the managed server and unload the model.

### `is_loaded -> bool`
Property indicating whether a model is loaded.

## Context Manager

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    print(engine.infer("Hello!").text)
```

## InferResult {#infer-result}

`InferResult` wraps inference output and metrics:

```python
result = engine.infer("Hello")
print(result.success)
print(result.text)
print(result.tokens_generated)
print(result.latency_ms)
print(result.tokens_per_sec)
print(result.error_message)
```
