# Unsloth Integration with llcuda

Learn how to fine-tune models with Unsloth and deploy them using llcuda for ultra-fast inference on Tesla T4 GPUs.

!!! tip "Complete Workflow"
    This guide covers the full pipeline: fine-tuning with Unsloth → exporting to GGUF → deploying with llcuda.

## Overview

[Unsloth](https://github.com/unslothai/unsloth) is a library for memory-efficient fine-tuning of large language models. Combined with llcuda, you get:

- **Fast fine-tuning:** 2x faster than standard methods with Unsloth
- **Memory efficient:** QLoRA with 4-bit quantization
- **Production deployment:** Export to GGUF and run with llcuda
- **Cost effective:** Train and deploy on free Google Colab T4

## Workflow Diagram

```mermaid
graph LR
    A[Base Model] --> B[Fine-tune with Unsloth]
    B --> C[Export to GGUF]
    C --> D[Deploy with llcuda]
    D --> E[Production Inference]
```

## Prerequisites

- Google Colab with T4 GPU
- Python 3.10+
- Basic understanding of fine-tuning
- Dataset for fine-tuning

## Step 1: Install Unsloth and llcuda

```python
# Install Unsloth (includes all dependencies)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install llcuda for deployment
!pip install git+https://github.com/waqasm86/llcuda.git
```

## Step 2: Fine-tune with Unsloth

### Load Base Model

```python
from unsloth import FastLanguageModel
import torch

# Load Gemma 3-1B for fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Use 4-bit quantization
)
```

### Configure LoRA

```python
# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but 0 is optimized
    bias="none",
    use_gradient_checkpointing="unsloth",  # Long context support
    random_state=3407,
)
```

### Prepare Dataset

```python
from datasets import load_dataset

# Load your dataset (example: alpaca format)
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Format prompts
def formatting_func(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)
```

### Train Model

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Start training
trainer.train()
```

## Step 3: Export to GGUF

### Save Model with Unsloth

```python
# Save fine-tuned model
model.save_pretrained("gemma-3-1b-custom")
tokenizer.save_pretrained("gemma-3-1b-custom")

# Merge LoRA weights (optional, for GGUF export)
model.save_pretrained_merged(
    "gemma-3-1b-merged",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit"
)
```

### Convert to GGUF Format

Unsloth provides built-in GGUF export:

```python
# Quantize and export to GGUF
model.save_pretrained_gguf(
    "gemma-3-1b-custom",
    tokenizer,
    quantization_method="q4_k_m",  # Recommended for T4
)

# This creates: gemma-3-1b-custom-Q4_K_M.gguf
```

**Available quantization methods:**

| Method | Size | Quality | Speed | Recommendation |
|--------|------|---------|-------|----------------|
| `q4_k_m` | Smallest | Good | Fastest | ✅ Best for T4 |
| `q5_k_m` | Medium | Better | Fast | Good balance |
| `q8_0` | Large | Best | Slower | High accuracy |
| `f16` | Largest | Perfect | Slowest | Development only |

### Alternative: Manual Conversion

If Unsloth export doesn't work, use llama.cpp tools:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert HuggingFace model to GGUF
python convert.py /path/to/gemma-3-1b-merged \
  --outfile gemma-3-1b-custom-f16.gguf \
  --outtype f16

# Quantize to Q4_K_M
./quantize gemma-3-1b-custom-f16.gguf \
  gemma-3-1b-custom-Q4_K_M.gguf \
  Q4_K_M
```

## Step 4: Deploy with llcuda

### Load GGUF Model

```python
import llcuda

# Create inference engine
engine = llcuda.InferenceEngine()

# Load your fine-tuned model
engine.load_model(
    "/content/gemma-3-1b-custom-Q4_K_M.gguf",
    gpu_layers=99,  # Full GPU offload
    ctx_size=2048,
    auto_start=True,
    verbose=True
)
```

### Run Inference

```python
# Test your fine-tuned model
result = engine.infer(
    prompt="### Instruction:\nWrite a poem about AI\n\n### Response:\n",
    max_tokens=100,
    temperature=0.7
)

print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

## Complete End-to-End Example

Here's a complete Colab notebook workflow:

```python
# Cell 1: Setup
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install git+https://github.com/waqasm86/llcuda.git

# Cell 2: Load and Fine-tune
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-2-2b-it-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        max_steps=60,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)
trainer.train()

# Cell 3: Export to GGUF
model.save_pretrained_gguf(
    "gemma-3-1b-alpaca",
    tokenizer,
    quantization_method="q4_k_m",
)

# Cell 4: Deploy with llcuda
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "gemma-3-1b-alpaca-Q4_K_M.gguf",
    auto_start=True
)

# Test
result = engine.infer(
    "Explain what a neural network is:",
    max_tokens=100
)
print(result.text)
print(f"{result.tokens_per_sec:.1f} tok/s on T4")
```

## Performance Comparison

| Stage | Tool | Time (T4) | Memory |
|-------|------|-----------|--------|
| Fine-tuning | Unsloth | ~10 min (1K samples) | 8 GB |
| Export to GGUF | Unsloth | ~2 min | 4 GB |
| Inference | llcuda | **134 tok/s** | 1.2 GB |
| Traditional Fine-tuning | Transformers | ~25 min | 14 GB |
| Traditional Inference | Transformers | 45 tok/s | 3.5 GB |

**Speedup:** 3x faster inference, 2.5x faster training!

## Best Practices

### 1. Choose Right Base Model

```python
# For Gemma models (recommended)
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"  # 2B parameters
model_name = "unsloth/gemma-3-1b-it-bnb-4bit"  # 1B parameters

# For Llama models
model_name = "unsloth/llama-3-8b-bnb-4bit"     # 8B parameters

# For Mistral models
model_name = "unsloth/mistral-7b-v0.2-bnb-4bit" # 7B parameters
```

### 2. Optimize LoRA Settings

```python
# Small models (1-3B) - higher rank
r = 16
lora_alpha = 16

# Large models (7B+) - lower rank to save memory
r = 8
lora_alpha = 16
```

### 3. Use Appropriate Quantization

For llcuda deployment on T4:

- **Q4_K_M:** Best balance (recommended)
- **Q5_K_M:** Better quality, slightly slower
- **Q4_0:** Smaller, lower quality
- **Q8_0:** Best quality, slower

### 4. Test Before Deployment

```python
# Test GGUF model before production
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True)

# Run test prompts
test_prompts = [
    "Test prompt 1",
    "Test prompt 2",
    "Test prompt 3"
]

for prompt in test_prompts:
    result = engine.infer(prompt, max_tokens=50)
    print(f"{prompt} -> {result.text}")
    assert result.success, f"Failed: {result.error_message}"
```

## Troubleshooting

### GGUF Export Fails

```python
# Try manual export with merged weights
model.save_pretrained_merged(
    "model-merged",
    tokenizer,
    save_method="merged_16bit"
)

# Then use llama.cpp convert.py (see Step 3)
```

### Out of Memory During Fine-tuning

```python
# Reduce batch size
per_device_train_batch_size = 1

# Increase gradient accumulation
gradient_accumulation_steps = 8

# Reduce sequence length
max_seq_length = 1024
```

### Slow Inference with llcuda

```python
# Increase GPU layers
gpu_layers = 99  # Full offload

# Reduce context size
ctx_size = 2048  # Or 1024 for faster

# Use Q4_K_M quantization
# Avoid F16 for production
```

## Advanced: Multi-Model Deployment

Deploy multiple fine-tuned models:

```python
import llcuda

# Load model 1
engine1 = llcuda.InferenceEngine(server_url="http://127.0.0.1:8090")
engine1.load_model("model1-Q4_K_M.gguf", auto_start=True)

# Load model 2 on different port
engine2 = llcuda.InferenceEngine(server_url="http://127.0.0.1:8091")
engine2.load_model("model2-Q4_K_M.gguf", auto_start=True)

# Use both models
result1 = engine1.infer("Prompt for model 1")
result2 = engine2.infer("Prompt for model 2")
```

## Production Deployment

For production use:

```python
# Save model to persistent storage
!cp gemma-3-1b-custom-Q4_K_M.gguf /path/to/models/

# Create deployment script
deployment_script = """
import llcuda

def deploy_model():
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "/path/to/models/gemma-3-1b-custom-Q4_K_M.gguf",
        gpu_layers=99,
        ctx_size=2048,
        auto_start=True,
        silent=True  # Suppress logs
    )
    return engine

engine = deploy_model()

# Your production code here
"""

with open("deploy.py", "w") as f:
    f.write(deployment_script)
```

## Next Steps

- [Performance Optimization](performance.md) - Tune inference parameters
- [Benchmarks](../performance/benchmarks.md) - Compare model performance
- [Model Selection](../guides/model-selection.md) - Choose base models
- [Executed Example](gemma-3-1b-executed.md) - See real results

## Resources

- **Unsloth:** [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **Unsloth Docs:** [docs.unsloth.ai](https://docs.unsloth.ai)
- **llama.cpp:** [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **GGUF Spec:** [gguf specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

!!! success "Complete Pipeline"
    You now have a complete pipeline for fine-tuning models with Unsloth and deploying them with llcuda for ultra-fast inference on Tesla T4!
