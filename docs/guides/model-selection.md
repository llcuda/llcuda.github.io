# Model Selection Guide

Choose the right model and quantization for your use case with llcuda v2.2.0.

## Quick Recommendations

### For Tesla T4 (15 GB)

| Priority | Model | Quantization | Speed | VRAM | Quality |
|----------|-------|--------------|-------|------|---------|
| **Speed** | Gemma 3-1B | Q4_K_M | 134 tok/s | 1.2 GB | Excellent |
| **Balance** | Llama 3.2-3B | Q4_K_M | 48 tok/s | 2.0 GB | Very good |
| **Quality** | Qwen 2.5-7B | Q4_K_M | 21 tok/s | 5.0 GB | Excellent |

### For Limited VRAM (< 8 GB)

| GPU VRAM | Recommended Model | Quantization | Expected Speed |
|----------|-------------------|--------------|----------------|
| 4 GB | Gemma 3-1B | Q4_0 | ~140 tok/s |
| 6 GB | Gemma 3-1B | Q4_K_M | ~134 tok/s |
| 8 GB | Llama 3.2-3B | Q4_K_M | ~48 tok/s |

---

## Model Size Comparison

### Performance vs Quality Trade-off

| Model Family | Size | Params | Tokens/sec (T4) | VRAM | Best For |
|--------------|------|--------|-----------------|------|----------|
| **Gemma 3** | 1B | 1.2B | **134** | 1.2 GB | Interactive apps, chatbots |
| **Llama 3.2** | 3B | 3.2B | **48** | 2.0 GB | Balanced performance |
| **Qwen 2.5** | 7B | 7.6B | **21** | 5.0 GB | Quality-focused tasks |
| **Llama 3.1** | 8B | 8.0B | **19** | 5.5 GB | Production quality |
| **Mistral** | 7B | 7.2B | **22** | 5.2 GB | Code generation |

### Detailed Comparison

#### 1B Models (Best for Speed)

**Gemma 3-1B-it**

- **Speed:** 134 tok/s (Q4_K_M)
- **VRAM:** 1.2 GB
- **Strengths:**
  - Fastest inference
  - Excellent for interactive chat
  - Low VRAM requirements
  - Good quality for size
- **Weaknesses:**
  - Limited reasoning on complex tasks
  - Shorter context understanding
- **Use Cases:**
  - Customer service chatbots
  - Quick Q&A systems
  - Real-time code assistance
  - Mobile/edge deployment

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)
```

#### 3B Models (Balanced)

**Llama 3.2-3B-Instruct**

- **Speed:** 48 tok/s (Q4_K_M)
- **VRAM:** 2.0 GB
- **Strengths:**
  - Good balance of speed/quality
  - Better reasoning than 1B
  - Handles complex instructions
  - Still fast enough for real-time
- **Weaknesses:**
  - 3x slower than 1B models
  - Higher VRAM usage
- **Use Cases:**
  - Content generation
  - Code explanation
  - Document summarization
  - Educational applications

```python
engine.load_model(
    "unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    silent=True
)
```

#### 7B Models (Quality-Focused)

**Qwen 2.5-7B-Instruct**

- **Speed:** 21 tok/s (Q4_K_M)
- **VRAM:** 5.0 GB
- **Strengths:**
  - Excellent quality
  - Strong reasoning abilities
  - Great for complex tasks
  - Multilingual support
- **Weaknesses:**
  - 6x slower than 1B
  - Requires 5+ GB VRAM
- **Use Cases:**
  - Research and analysis
  - Complex reasoning tasks
  - Technical documentation
  - Multi-step problem solving

```python
engine.load_model(
    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    silent=True
)
```

**Llama 3.1-8B-Instruct**

- **Speed:** 19 tok/s (Q4_K_M)
- **VRAM:** 5.5 GB
- **Strengths:**
  - State-of-the-art quality
  - Excellent instruction following
  - Long context support (128K)
  - Multilingual
- **Use Cases:**
  - Production applications
  - API services
  - Complex workflows
  - Enterprise deployments

```python
engine.load_model(
    "unsloth/Llama-3.1-8B-Instruct-Q4_K_M-GGUF",
    silent=True
)
```

---

## Quantization Guide

### Understanding Quantization Types

| Quantization | Bits | Speed | Quality | VRAM | File Size | Recommendation |
|--------------|------|-------|---------|------|-----------|----------------|
| Q2_K | 2.5 | Fastest | 85% | Lowest | ~30% | Prototyping only |
| Q3_K_M | 3.5 | Very fast | 92% | Very low | ~40% | Emergency low VRAM |
| Q4_0 | 4.0 | Fast | 97% | Low | ~45% | Speed priority |
| **Q4_K_M** | 4.5 | **Fast** | **99%** | **Medium** | **~50%** | ✅ **Recommended** |
| Q5_K_M | 5.5 | Moderate | 99.5% | Medium-high | ~60% | Quality critical |
| Q6_K | 6.5 | Slow | 99.8% | High | ~70% | Rarely needed |
| Q8_0 | 8.0 | Slower | 99.95% | Very high | ~85% | Development only |
| F16 | 16.0 | Slowest | 100% | Maximum | 100% | Not recommended |

### Choosing Quantization

**For most users:**
```python
# Q4_K_M: Best overall choice
engine.load_model(
    "model-Q4_K_M.gguf",
    silent=True
)
```

**For speed-critical applications:**
```python
# Q4_0: 3-5% faster, slightly lower quality
engine.load_model(
    "model-Q4_0.gguf",
    silent=True
)
```

**For quality-critical work:**
```python
# Q5_K_M: Better quality, 20% slower
engine.load_model(
    "model-Q5_K_M.gguf",
    silent=True
)
```

**For extreme VRAM constraints:**
```python
# Q3_K_M: Smallest usable quantization
engine.load_model(
    "model-Q3_K_M.gguf",
    silent=True
)
```

---

## Popular Model Collections

### Unsloth Models (Recommended)

Unsloth provides optimized GGUF models on HuggingFace:

**Gemma Models:**
```python
# Gemma 3-1B (Best for speed)
"unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"

# Gemma 2-2B
"unsloth/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K_M.gguf"

# Gemma 2-9B
"unsloth/gemma-2-9b-it-GGUF:gemma-2-9b-it-Q4_K_M.gguf"
```

**Llama Models:**
```python
# Llama 3.2-1B
"unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Llama 3.2-3B
"unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Llama 3.1-8B
"unsloth/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

**Mistral Models:**
```python
# Mistral 7B v0.3
"unsloth/Mistral-7B-Instruct-v0.3-GGUF:Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

# Mistral Nemo 12B
"unsloth/Mistral-Nemo-Instruct-2407-GGUF:Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
```

### Official HuggingFace Models

**Qwen Models:**
```python
# Qwen 2.5-7B (Excellent quality)
"Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q4_k_m.gguf"

# Qwen 2.5-14B
"Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m.gguf"
```

**Phi Models:**
```python
# Phi 3.5-Mini (3.8B)
"microsoft/Phi-3.5-mini-instruct-gguf:Phi-3.5-mini-instruct-Q4_K_M.gguf"
```

---

## VRAM Requirements

### Model Size to VRAM Mapping

For Q4_K_M quantization:

| Model Size | Q4_K_M VRAM | Q5_K_M VRAM | Q8_0 VRAM | ctx=2048 |
|------------|-------------|-------------|-----------|----------|
| 1B | 1.2 GB | 1.5 GB | 2.5 GB | Add +0.3 GB |
| 3B | 2.0 GB | 2.4 GB | 4.2 GB | Add +0.3 GB |
| 7B | 5.0 GB | 6.2 GB | 9.5 GB | Add +0.5 GB |
| 8B | 5.5 GB | 6.8 GB | 10.2 GB | Add +0.5 GB |
| 13B | 9.0 GB | 11.0 GB | 16.5 GB | Add +0.8 GB |

### GPU Recommendations

| GPU | VRAM | Max Model (Q4_K_M) | Recommended Model |
|-----|------|-------------------|-------------------|
| **Tesla T4** | 15 GB | 7B | 1B (speed) or 7B (quality) |
| RTX 3060 | 12 GB | 7B | 3B |
| RTX 3070 | 8 GB | 3B | 1B |
| RTX 3080 | 10 GB | 7B | 3B |
| RTX 3090 | 24 GB | 13B | 7B |
| RTX 4070 | 12 GB | 7B | 3B |
| RTX 4090 | 24 GB | 13B | 7B or 13B |
| A100 | 40 GB | 30B | 13B |
| A100 | 80 GB | 70B | 30B |

---

## Use Case Recommendations

### Interactive Chatbots

**Priority:** Speed, low latency

**Recommended:**
- Gemma 3-1B Q4_K_M (134 tok/s)
- Llama 3.2-1B Q4_K_M (140 tok/s)

```python
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    ctx_size=2048,
    silent=True
)
```

### Code Generation

**Priority:** Accuracy, context understanding

**Recommended:**
- Qwen 2.5-7B Q4_K_M (21 tok/s)
- Llama 3.1-8B Q4_K_M (19 tok/s)

```python
engine.load_model(
    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    ctx_size=4096,  # Longer context for code
    silent=True
)
```

### Document Summarization

**Priority:** Context length, quality

**Recommended:**
- Llama 3.1-8B Q4_K_M (128K context)
- Qwen 2.5-7B Q4_K_M

```python
engine.load_model(
    "unsloth/Llama-3.1-8B-Instruct-Q4_K_M-GGUF",
    ctx_size=8192,  # Long documents
    silent=True
)
```

### Question Answering

**Priority:** Accuracy, speed

**Recommended:**
- Llama 3.2-3B Q4_K_M (48 tok/s)
- Gemma 3-1B Q4_K_M (134 tok/s)

```python
engine.load_model(
    "unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    ctx_size=2048,
    silent=True
)
```

### Content Generation

**Priority:** Creativity, quality

**Recommended:**
- Qwen 2.5-7B Q5_K_M
- Llama 3.1-8B Q5_K_M

```python
engine.load_model(
    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q5_K_M",
    ctx_size=4096,
    silent=True
)

# Use creative generation settings
result = engine.infer(
    prompt,
    temperature=1.0,
    top_p=0.95,
    max_tokens=500
)
```

### Education & Tutoring

**Priority:** Accuracy, explanations

**Recommended:**
- Llama 3.2-3B Q4_K_M
- Qwen 2.5-7B Q4_K_M

```python
engine.load_model(
    "unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    ctx_size=2048,
    silent=True
)
```

---

## Model Capabilities

### Multilingual Support

| Model | Languages | Notes |
|-------|-----------|-------|
| Gemma 3-1B | English primarily | Limited multilingual |
| Llama 3.2-3B | 8 languages | Good multilingual |
| Llama 3.1-8B | 8 languages | Excellent multilingual |
| Qwen 2.5-7B | 29 languages | Best multilingual |
| Mistral 7B | English, French, German, Spanish, Italian | Good European languages |

### Context Window Support

| Model | Standard Context | Max Context | Notes |
|-------|------------------|-------------|-------|
| Gemma 3-1B | 2K | 8K | Limited long context |
| Llama 3.2-3B | 4K | 128K | Excellent long context |
| Llama 3.1-8B | 8K | 128K | Best long context |
| Qwen 2.5-7B | 8K | 32K | Good long context |
| Mistral 7B | 8K | 32K | Good long context |

### Special Capabilities

| Model | Code | Math | Reasoning | Function Calling |
|-------|------|------|-----------|------------------|
| Gemma 3-1B | Good | Fair | Fair | No |
| Llama 3.2-3B | Very Good | Good | Good | Yes |
| Llama 3.1-8B | Excellent | Very Good | Excellent | Yes |
| Qwen 2.5-7B | Excellent | Excellent | Excellent | Yes |
| Mistral 7B | Very Good | Good | Good | Yes |

---

## Finding and Loading Models

### From Unsloth (Recommended)

```python
# Browse models at: https://huggingface.co/unsloth
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)
```

### From Official Repos

```python
# Qwen
engine.load_model(
    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    silent=True
)

# Microsoft Phi
engine.load_model(
    "microsoft/Phi-3.5-mini-instruct-gguf:Phi-3.5-mini-instruct-Q4_K_M.gguf",
    silent=True
)
```

### Local Models

```python
# Load from local path
engine.load_model(
    "/path/to/model.gguf",
    silent=True
)
```

---

## Model Evaluation

### Quick Quality Test

```python
import llcuda

def evaluate_model(model_path):
    """Quick quality evaluation."""

    engine = llcuda.InferenceEngine()
    engine.load_model(model_path, silent=True)

    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
        "What are the causes of climate change?",
        "Translate 'Hello, how are you?' to Spanish.",
        "Solve: If x + 5 = 12, what is x?"
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}\n")

    for i, prompt in enumerate(test_prompts, 1):
        result = engine.infer(prompt, max_tokens=150)

        print(f"{i}. {prompt}")
        print(f"   Response: {result.text[:100]}...")
        print(f"   Speed: {result.tokens_per_sec:.1f} tok/s\n")

    metrics = engine.get_metrics()
    print(f"Average speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
    print(f"Average latency: {metrics['latency']['mean_ms']:.0f}ms")

# Test multiple models
models = [
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    "unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
]

for model in models:
    evaluate_model(model)
```

---

## Migration Guide

### From Larger to Smaller Models

If you need to reduce VRAM:

```python
# Before: 7B model (5 GB VRAM)
engine.load_model(
    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    silent=True
)

# After: 3B model (2 GB VRAM)
engine.load_model(
    "unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    silent=True
)
```

### From Higher to Lower Quantization

```python
# Before: Q5_K_M (better quality, slower)
engine.load_model("model-Q5_K_M.gguf", silent=True)

# After: Q4_K_M (faster, minimal quality loss)
engine.load_model("model-Q4_K_M.gguf", silent=True)
```

---

## See Also

- [GGUF Format](gguf-format.md) - Understanding GGUF
- [Performance Benchmarks](../performance/benchmarks.md) - Speed comparisons
- [Optimization Guide](../performance/optimization.md) - Tuning performance
- [Quick Start](quickstart.md) - Getting started
- [HuggingFace Models](https://huggingface.co/models?library=gguf) - Browse GGUF models
