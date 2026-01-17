# Memory Management

Manage VRAM efficiently on dual T4.

## VRAM Budget

**Total**: 30 GB (2× 15GB T4)
**Usable**: ~28 GB (leave headroom)

## Allocation Strategy

### Single GPU (15GB)
- Model: 8-12 GB
- KV Cache: 2-4 GB
- Overhead: 1-2 GB

### Dual GPU (30GB)
- Model: 20-26 GB
- KV Cache: 2-4 GB
- Overhead: 2-3 GB

## Reduce Memory Usage

1. **Smaller Quantization**
   - Q4_K_M → IQ3_XS

2. **Smaller Context**
   - 8192 → 2048 tokens

3. **Smaller Batch**
   - 2048 → 512

4. **No KV Offload**
   ```python
   config = ServerConfig(no_kv_offload=True)
   ```
