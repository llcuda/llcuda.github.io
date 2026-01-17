# GGUF Tools API

GGUF file parsing and utilities.

## Overview

Tools for working with GGUF model files.

## Basic Usage

```python
from llcuda.utils import GGUFParser

parser = GGUFParser(model_path="model.gguf")
print(f"Parameters: {parser.get_parameter_count() / 1e9:.1f}B")
print(f"Quantization: {parser.get_quantization()}")
```

## Class Reference

### `GGUFParser`

```python
class GGUFParser:
    def __init__(self, model_path: str):
        """Initialize parser.
        
        Args:
            model_path: Path to GGUF file
        """
        
    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        
    def get_quantization(self) -> str:
        """Get quantization type."""
        
    def get_context_length(self) -> int:
        """Get max context length."""
        
    def get_metadata(self) -> Dict:
        """Get all metadata."""
```

## Functions

### `estimate_vram()`

```python
def estimate_vram(model_size_b: float, quant_type: str) -> float:
    """Estimate VRAM usage.
    
    Args:
        model_size_b: Model size in billions
        quant_type: Quantization type (Q4_K_M, IQ3_XS, etc.)
        
    Returns:
        Estimated VRAM in GB
    """
```

## Quantization Types

**K-Quants**:
- Q4_K_M, Q5_K_M, Q6_K, Q8_0

**I-Quants**:
- IQ3_XS, IQ2_XXS

## Examples

See [GGUF Tutorial](../tutorials/04-gguf-quantization.md)
