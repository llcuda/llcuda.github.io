# GGUF Export

Export Unsloth models to GGUF format.

## Basic Export

```python
model.save_pretrained_gguf(
    "output_dir",
    tokenizer,
    quantization_method="q4_k_m",
)
```

## Quantization Options

| Method | Size | Quality | Use Case |
|--------|------|---------|----------|
| `q4_k_m` | 4.8 bpw | Good | Recommended |
| `q5_k_m` | 5.7 bpw | Better | Higher quality |
| `q8_0` | 8.5 bpw | Excellent | Maximum quality |

## Advanced Options

```python
model.save_pretrained_gguf(
    "output_dir",
    tokenizer,
    quantization_method="q4_k_m",
    
    # Optional
    push_to_hub=False,
    token=None,
    save_method="merged_16bit",  # or "lora"
)
```

## Output Files

```
output_dir/
├── my_model-Q4_K_M.gguf      # Quantized model
├── my_model-F16.gguf         # Optional: FP16 version
└── config.json               # Model config
```
