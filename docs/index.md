# llcuda v2.0.6: Tesla T4 CUDA Inference

<div style="text-align: center; margin: 2em 0;">
  <img src="https://img.shields.io/badge/version-2.0.6-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.11+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.x-orange.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/GPU-Tesla%20T4-green.svg" alt="GPU">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</div>

**Fast LLM inference on Tesla T4 GPUs with FlashAttention and Tensor Core optimization.** Built exclusively for Google Colab and Tesla T4 hardware with GitHub-only distribution.

## :rocket: Why llcuda v2.0.6?

=== "Tesla T4 Optimized"
    Built specifically for Tesla T4 (SM 7.5) with:

    - ✅ FlashAttention support (2-3x faster)
    - ✅ Tensor Core optimization
    - ✅ CUDA Graphs for reduced overhead
    - ✅ **134 tokens/sec verified** on Gemma 3-1B

=== "GitHub-Only Distribution"
    No PyPI dependency:

    ```bash
    pip install git+https://github.com/waqasm86/llcuda.git
    ```

    - Binaries auto-download from GitHub Releases (266 MB)
    - One-time setup, cached for future use
    - Direct from source, always up-to-date

=== "Google Colab Ready"
    Perfect for cloud notebooks:

    - ✅ Tesla T4 Free tier supported
    - ✅ One-line install
    - ✅ Instant inference
    - ✅ Verified 134 tok/s performance

=== "Unsloth Integration"
    Seamless workflow:

    - Fine-tune with Unsloth (2x faster training)
    - Export to GGUF format
    - Deploy with llcuda (fast inference)
    - Production-ready pipeline

## :fire: Quick Start

Try llcuda on Google Colab right now!

<div style="text-align: center; margin: 2em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_v2_0_6_gemma3_1b_unsloth_colab.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="width: 200px;">
  </a>
</div>

### 60-Second Setup

```python
# Install from GitHub
pip install git+https://github.com/waqasm86/llcuda.git

# Run inference
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

result = engine.infer(
    "Explain quantum computing in simple terms",
    max_tokens=200
)

print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
# Expected output: ~134 tokens/sec on Tesla T4
```

!!! success "First Run Downloads"
    CUDA binaries (266 MB) download automatically from GitHub Releases v2.0.6 on first import.
    Subsequent runs use cached binaries - instant startup!

## :chart_with_upwards_trend: Verified Performance

Real Google Colab Tesla T4 results with **proven 3x faster performance**:

| Model | Quantization | Speed | Latency | VRAM | Status |
|-------|--------------|-------|---------|------|--------|
| **Gemma 3-1B** | Q4_K_M | **134 tok/s** | 690ms | 1.2 GB | ✅ Verified |
| Llama 3.2-3B | Q4_K_M | ~30 tok/s | - | 2.0 GB | Estimated |
| Qwen 2.5-7B | Q4_K_M | ~18 tok/s | - | 5.0 GB | Estimated |
| Llama 3.1-8B | Q4_K_M | ~15 tok/s | - | 5.5 GB | Estimated |

!!! tip "Performance Highlights"
    - **3x faster than expected** (134 vs 45 tok/s initial estimate)
    - Consistent 130-142 tok/s range across batch inference
    - Full GPU offload (99 layers on T4)
    - FlashAttention + Tensor Cores delivering exceptional results

[:octicons-file-code-24: See Executed Notebook](tutorials/gemma-3-1b-executed.md){ .md-button .md-button--primary }

## :sparkles: Features

<div class="grid cards" markdown>

-   :material-rocket-launch: **Auto-Download**

    ---

    Fetch CUDA binaries and GGUF models automatically

    - GitHub Releases integration
    - HuggingFace model support
    - Smart caching system

-   :material-speedometer: **Optimized for T4**

    ---

    Built specifically for Tesla T4 GPUs

    - SM 7.5 targeting
    - FlashAttention enabled
    - Tensor Core support

-   :material-chat: **Easy API**

    ---

    PyTorch-style inference interface

    - Single-line model loading
    - Batch processing
    - Streaming support

-   :material-cloud-upload: **Production Ready**

    ---

    Reliable and well-tested

    - Comprehensive error handling
    - Silent mode for servers
    - MIT licensed

</div>

## :books: Use Cases

=== "Interactive Chat"
    ```python
    import llcuda

    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = engine.infer(user_input, max_tokens=400)
        print(f"Assistant: {result.text}")
    ```

=== "Batch Processing"
    ```python
    import llcuda

    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    prompts = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "Define deep learning concisely."
    ]

    results = engine.batch_infer(prompts, max_tokens=80)

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result.text}")
        print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")
    ```

=== "Google Colab"
    ```python
    import llcuda

    # Verify GPU compatibility
    compat = llcuda.check_gpu_compatibility()
    print(f"GPU: {compat['gpu_name']}")
    print(f"Compatible: {compat['compatible']}")

    # Load model
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    # Run inference
    result = engine.infer(
        "Explain artificial intelligence",
        max_tokens=300
    )
    print(result.text)
    print(f"Performance: {result.tokens_per_sec:.1f} tok/s")
    ```

=== "Unsloth Workflow"
    ```python
    # Step 1: Fine-tune with Unsloth
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/gemma-3-1b-it",
        max_seq_length=2048,
        load_in_4bit=True
    )

    # Train your model...

    # Step 2: Export to GGUF
    model.save_pretrained_gguf(
        "my_model",
        tokenizer,
        quantization_method="q4_k_m"
    )

    # Step 3: Deploy with llcuda
    import llcuda

    engine = llcuda.InferenceEngine()
    engine.load_model("my_model/unsloth.Q4_K_M.gguf")

    result = engine.infer("Your prompt", max_tokens=200)
    print(result.text)
    ```

## :books: Next Steps

<div class="grid cards" markdown>

- [:material-rocket-launch: **Quick Start Guide**](guides/quickstart.md)

    Get started in 5 minutes with step-by-step instructions

- [:material-download: **Installation**](guides/installation.md)

    Detailed installation for Google Colab and local systems

- [:material-google: **Google Colab Tutorial**](tutorials/gemma-3-1b-colab.md)

    Complete walkthrough with Tesla T4 GPU examples

- [:material-code-braces: **API Reference**](api/overview.md)

    Full API documentation and advanced usage

- [:material-chart-line: **Performance Benchmarks**](performance/benchmarks.md)

    Detailed benchmarks and optimization tips

- [:material-notebook: **Jupyter Notebooks**](notebooks/index.md)

    Ready-to-run Colab notebooks with examples

</div>

## :sparkles: What's New in v2.0.6

- **GitHub-Only Distribution** - Removed PyPI dependency completely
- **Verified Performance** - Real Tesla T4 results: **134 tok/s** on Gemma 3-1B
- **Updated Bootstrap** - Auto-download from GitHub Releases v2.0.6
- **Comprehensive Tutorials** - New Colab notebooks with live execution outputs
- **Enhanced Documentation** - Complete guides and API reference
- **Same Proven Binaries** - Uses stable v2.0.3 CUDA binaries (identical SHA256)

[:octicons-arrow-right-24: Read Changelog](https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md){ .md-button }

## :handshake: Community & Support

- **GitHub Repository**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **GitHub Releases**: [Releases & Downloads](https://github.com/waqasm86/llcuda/releases)
- **Bug Reports**: [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
- **Email**: waqasm86@gmail.com

## :balance_scale: License

MIT License - Free for commercial and personal use.

---

<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 3em;">
  Built with ❤️ by <a href="https://github.com/waqasm86">Waqas Muhammad</a> | Powered by <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> | Optimized for <a href="https://github.com/unslothai/unsloth">Unsloth</a>
</div>
