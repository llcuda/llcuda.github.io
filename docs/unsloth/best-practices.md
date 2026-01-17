# Best Practices

Recommendations for Unsloth + llcuda workflow.

## Model Selection

### For Single T4 (15GB)
- Qwen2.5-1.5B
- Gemma 2-2B
- Llama-3.2-3B

### For Dual T4 (30GB)
- Qwen2.5-7B
- Llama-3.1-8B
- Mistral-7B

## Quantization

| Model Size | Training | Export |
|------------|----------|--------|
| 1-3B | 4-bit QLoRA | Q4_K_M |
| 7-8B | 4-bit QLoRA | Q4_K_M |
| 13B+ | 4-bit QLoRA | IQ3_XS |

## Training Tips

1. **Use QLoRA** (4-bit)
   - 70% less VRAM
   - 2x faster training

2. **Optimal LoRA rank**
   - Small models: r=8-16
   - Large models: r=16-32

3. **Gradient checkpointing**
   - Reduces memory
   - Slightly slower

## Deployment Tips

1. **Enable FlashAttention**
2. **Use tensor-split for large models**
3. **Monitor VRAM usage**
4. **Test with small batches first**
