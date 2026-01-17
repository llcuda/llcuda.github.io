# Fine-Tuning Workflow

Fine-tune models with Unsloth for llcuda deployment.

## Step 1: Setup Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,  # QLoRA
)
```

## Step 2: Add LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
)
```

## Step 3: Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

## Step 4: Export GGUF

```python
model.save_pretrained_gguf(
    "my_finetuned_model",
    tokenizer,
    quantization_method="q4_k_m",  # Recommended for T4
)
```

Output: `my_finetuned_model-Q4_K_M.gguf`

See: [Tutorial 05](../tutorials/05-unsloth-integration.md)
