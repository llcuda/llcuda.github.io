# Code Examples

Complete, production-ready code examples for common llcuda use cases.

## Quick Reference

| Example | Use Case | Complexity |
|---------|----------|------------|
| [Basic Inference](#basic-inference) | Single question-answer | Beginner |
| [Chat Application](#chat-application) | Interactive conversation | Beginner |
| [Batch Processing](#batch-processing) | Process multiple prompts | Beginner |
| [Streaming Inference](#streaming-inference) | Real-time token generation | Intermediate |
| [Custom Parameters](#custom-generation-parameters) | Fine-tune generation | Intermediate |
| [Context Manager](#context-manager-pattern) | Auto-cleanup resources | Intermediate |
| [Error Handling](#robust-error-handling) | Production-ready code | Advanced |
| [Benchmarking](#performance-benchmarking) | Measure performance | Advanced |

---

## Basic Inference

Simple question-answer inference.

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Load model
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Run inference
result = engine.infer(
    "Explain quantum computing in simple terms",
    max_tokens=200,
    temperature=0.7
)

# Print results
print(f"Response: {result.text}")
print(f"\nPerformance:")
print(f"  Speed: {result.tokens_per_sec:.1f} tokens/sec")
print(f"  Latency: {result.latency_ms:.0f}ms")
print(f"  Tokens: {result.tokens_generated}")
```

**Expected Output on Tesla T4:**
```
Response: Quantum computing uses quantum mechanics principles...

Performance:
  Speed: 134.2 tokens/sec
  Latency: 690ms
  Tokens: 93
```

---

## Chat Application

Interactive chat with conversation loop.

```python
import llcuda

def chat_application():
    """Interactive chat application with Gemma 3-1B."""

    # Initialize engine
    engine = llcuda.InferenceEngine()

    print("Loading Gemma 3-1B model...")
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    print("\n🤖 Chat with Gemma 3-1B")
    print("Type 'exit' to quit, 'clear' to reset metrics\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Handle commands
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break

        if user_input.lower() == 'clear':
            engine.reset_metrics()
            print("✅ Metrics reset\n")
            continue

        if not user_input:
            continue

        # Generate response
        result = engine.infer(
            user_input,
            max_tokens=300,
            temperature=0.7
        )

        # Display response
        print(f"\n🤖 AI: {result.text}")
        print(f"   ({result.tokens_per_sec:.1f} tok/s, {result.latency_ms:.0f}ms)\n")

    # Show final metrics
    metrics = engine.get_metrics()
    print("\n📊 Session Statistics:")
    print(f"  Total requests: {metrics['throughput']['total_requests']}")
    print(f"  Total tokens: {metrics['throughput']['total_tokens']}")
    print(f"  Avg speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
    print(f"  Avg latency: {metrics['latency']['mean_ms']:.0f}ms")

# Run the chat app
if __name__ == "__main__":
    chat_application()
```

**Sample Interaction:**
```
You: What is machine learning?

🤖 AI: Machine learning is a subset of artificial intelligence that enables
   computers to learn from data without explicit programming...
   (134.5 tok/s, 685ms)

You: Give me an example

🤖 AI: A common example is email spam filtering. The system learns to
   identify spam by analyzing thousands of emails...
   (136.2 tok/s, 702ms)

You: exit

📊 Session Statistics:
  Total requests: 2
  Total tokens: 184
  Avg speed: 135.2 tok/s
  Avg latency: 694ms
```

---

## Batch Processing

Process multiple prompts efficiently.

```python
import llcuda
import time

def batch_processing_example():
    """Process multiple prompts with performance tracking."""

    # Initialize engine
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    # Define prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain neural networks briefly.",
        "What is deep learning?",
        "Define machine learning.",
        "What are transformers in AI?",
        "Explain backpropagation.",
        "What is gradient descent?",
        "Define overfitting in ML."
    ]

    print(f"Processing {len(prompts)} prompts...\n")

    # Reset metrics
    engine.reset_metrics()

    # Process batch
    start_time = time.time()
    results = engine.batch_infer(prompts, max_tokens=80, temperature=0.7)
    total_time = time.time() - start_time

    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"{i}. Q: {prompt}")
        print(f"   A: {result.text[:100]}...")
        print(f"   Performance: {result.tokens_per_sec:.1f} tok/s, {result.latency_ms:.0f}ms\n")

    # Show aggregate metrics
    metrics = engine.get_metrics()
    print("📊 Batch Processing Summary:")
    print(f"  Prompts processed: {len(prompts)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {metrics['throughput']['total_tokens']}")
    print(f"  Avg throughput: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
    print(f"  Avg latency: {metrics['latency']['mean_ms']:.0f}ms")
    print(f"  P95 latency: {metrics['latency']['p95_ms']:.0f}ms")
    print(f"  Requests/sec: {len(prompts) / total_time:.2f}")

# Run batch processing
if __name__ == "__main__":
    batch_processing_example()
```

**Expected Output:**
```
Processing 8 prompts...

1. Q: What is artificial intelligence?
   A: Artificial intelligence (AI) is the simulation of human intelligence...
   Performance: 134.8 tok/s, 685ms

2. Q: Explain neural networks briefly.
   A: Neural networks are computational models inspired by the human brain...
   Performance: 135.2 tok/s, 692ms

[...]

📊 Batch Processing Summary:
  Prompts processed: 8
  Total time: 5.52s
  Total tokens: 592
  Avg throughput: 134.5 tok/s
  Avg latency: 690ms
  P95 latency: 725ms
  Requests/sec: 1.45
```

---

## Streaming Inference

Stream tokens as they're generated (simulation).

```python
import llcuda
import time

def streaming_inference_example():
    """Demonstrate streaming inference with callback."""

    # Initialize engine
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    # Define callback for streaming
    def stream_callback(chunk):
        """Print each chunk as it arrives."""
        print(chunk, end='', flush=True)

    prompt = "Write a short story about a robot learning to paint"

    print("🤖 Generating story (streaming):\n")
    print("AI: ", end='', flush=True)

    # Stream inference
    result = engine.infer_stream(
        prompt,
        callback=stream_callback,
        max_tokens=200,
        temperature=0.8
    )

    # Show metrics
    print(f"\n\n📊 Performance:")
    print(f"  Speed: {result.tokens_per_sec:.1f} tok/s")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Tokens: {result.tokens_generated}")

# Run streaming example
if __name__ == "__main__":
    streaming_inference_example()
```

**Note:** Current implementation simulates streaming. True token-by-token streaming will be available in a future release.

---

## Custom Generation Parameters

Fine-tune generation with custom parameters.

```python
import llcuda

def custom_parameters_example():
    """Demonstrate different generation strategies."""

    # Initialize engine
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    prompt = "Once upon a time in a futuristic city"

    # Strategy 1: Deterministic (low temperature)
    print("1️⃣ Deterministic Generation (temp=0.1):")
    result1 = engine.infer(
        prompt,
        max_tokens=100,
        temperature=0.1,
        top_p=0.9,
        top_k=10
    )
    print(f"{result1.text}\n")

    # Strategy 2: Balanced (default)
    print("2️⃣ Balanced Generation (temp=0.7):")
    result2 = engine.infer(
        prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )
    print(f"{result2.text}\n")

    # Strategy 3: Creative (high temperature)
    print("3️⃣ Creative Generation (temp=1.2):")
    result3 = engine.infer(
        prompt,
        max_tokens=100,
        temperature=1.2,
        top_p=0.95,
        top_k=100
    )
    print(f"{result3.text}\n")

    # Strategy 4: Very creative (high temp + nucleus sampling)
    print("4️⃣ Very Creative (temp=1.5, top_p=0.95):")
    result4 = engine.infer(
        prompt,
        max_tokens=100,
        temperature=1.5,
        top_p=0.95,
        top_k=200
    )
    print(f"{result4.text}\n")

    # Compare performance
    print("📊 Performance Comparison:")
    for i, result in enumerate([result1, result2, result3, result4], 1):
        print(f"  Strategy {i}: {result.tokens_per_sec:.1f} tok/s")

# Run custom parameters example
if __name__ == "__main__":
    custom_parameters_example()
```

**Parameter Guide:**

| Parameter | Range | Effect | Use Case |
|-----------|-------|--------|----------|
| `temperature` | 0.1 - 0.3 | Deterministic, focused | Code, facts |
| `temperature` | 0.6 - 0.8 | Balanced creativity | General chat |
| `temperature` | 1.0 - 1.5 | Very creative | Stories, brainstorming |
| `top_p` | 0.9 - 0.95 | Nucleus sampling | Quality control |
| `top_k` | 10 - 200 | Diversity limit | Token variety |

---

## Context Manager Pattern

Automatic resource cleanup.

```python
import llcuda

def context_manager_example():
    """Use context manager for automatic cleanup."""

    # Context manager ensures server cleanup
    with llcuda.InferenceEngine() as engine:
        # Load model
        engine.load_model(
            "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
            silent=True
        )

        # Run inferences
        prompts = [
            "What is Python?",
            "What is JavaScript?",
            "What is Rust?"
        ]

        for prompt in prompts:
            result = engine.infer(prompt, max_tokens=50)
            print(f"Q: {prompt}")
            print(f"A: {result.text}\n")

        # Get final metrics
        metrics = engine.get_metrics()
        print(f"Total tokens: {metrics['throughput']['total_tokens']}")

    # Server automatically stopped here
    print("✅ Server cleaned up automatically")

# Run context manager example
if __name__ == "__main__":
    context_manager_example()
```

---

## Robust Error Handling

Production-ready error handling.

```python
import llcuda
from llcuda import InferenceEngine

def robust_inference(prompt: str, max_retries: int = 3):
    """Robust inference with error handling and retries."""

    engine = None

    try:
        # Check GPU compatibility
        compat = llcuda.check_gpu_compatibility()
        if not compat['compatible']:
            raise RuntimeError(
                f"GPU {compat['gpu_name']} is not compatible: {compat['reason']}"
            )

        # Initialize engine
        engine = InferenceEngine()

        # Load model with error handling
        try:
            engine.load_model(
                "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
                silent=True,
                auto_start=True
            )
        except FileNotFoundError as e:
            print(f"Model not found: {e}")
            print("Please download the model first")
            return None
        except RuntimeError as e:
            print(f"Server failed to start: {e}")
            return None

        # Run inference with retries
        for attempt in range(max_retries):
            result = engine.infer(prompt, max_tokens=200)

            if result.success:
                return {
                    'text': result.text,
                    'tokens_per_sec': result.tokens_per_sec,
                    'latency_ms': result.latency_ms,
                    'success': True
                }
            else:
                print(f"Attempt {attempt + 1} failed: {result.error_message}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({max_retries - attempt - 1} attempts left)")
                    import time
                    time.sleep(1)

        # All retries failed
        return {
            'text': None,
            'error': 'All retry attempts failed',
            'success': False
        }

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            'text': None,
            'error': str(e),
            'success': False
        }

    finally:
        # Cleanup
        if engine is not None:
            engine.unload_model()

# Example usage
if __name__ == "__main__":
    result = robust_inference("What is quantum computing?")

    if result and result['success']:
        print(f"✅ Success!")
        print(f"Response: {result['text']}")
        print(f"Speed: {result['tokens_per_sec']:.1f} tok/s")
    else:
        print(f"❌ Failed: {result['error'] if result else 'Unknown error'}")
```

---

## Performance Benchmarking

Comprehensive performance measurement.

```python
import llcuda
import time
import statistics

def benchmark_inference(num_runs: int = 10):
    """Benchmark inference performance."""

    # Initialize
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    # Warmup
    print("Warming up...")
    for _ in range(3):
        engine.infer("Warmup prompt", max_tokens=10)

    # Benchmark
    print(f"Running {num_runs} iterations...\n")

    engine.reset_metrics()
    latencies = []
    throughputs = []

    test_prompt = "Explain the concept of recursion in programming"

    for i in range(num_runs):
        start = time.time()
        result = engine.infer(test_prompt, max_tokens=100)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        latencies.append(result.latency_ms)
        throughputs.append(result.tokens_per_sec)

        print(f"Run {i+1}/{num_runs}: "
              f"{result.tokens_per_sec:.1f} tok/s, "
              f"{result.latency_ms:.0f}ms")

    # Calculate statistics
    print("\n" + "="*60)
    print("📊 Benchmark Results")
    print("="*60)

    print("\nThroughput (tokens/sec):")
    print(f"  Mean:   {statistics.mean(throughputs):.2f}")
    print(f"  Median: {statistics.median(throughputs):.2f}")
    print(f"  Stdev:  {statistics.stdev(throughputs):.2f}")
    print(f"  Min:    {min(throughputs):.2f}")
    print(f"  Max:    {max(throughputs):.2f}")

    print("\nLatency (ms):")
    print(f"  Mean:   {statistics.mean(latencies):.2f}")
    print(f"  Median: {statistics.median(latencies):.2f}")
    print(f"  Stdev:  {statistics.stdev(latencies):.2f}")
    print(f"  Min:    {min(latencies):.2f}")
    print(f"  Max:    {max(latencies):.2f}")

    # Percentiles
    sorted_latencies = sorted(latencies)
    p50_idx = len(sorted_latencies) // 2
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)

    print("\nLatency Percentiles:")
    print(f"  P50:    {sorted_latencies[p50_idx]:.2f}ms")
    print(f"  P95:    {sorted_latencies[p95_idx]:.2f}ms")
    print(f"  P99:    {sorted_latencies[p99_idx]:.2f}ms")

    # Get metrics from engine
    metrics = engine.get_metrics()
    print(f"\nTotal tokens generated: {metrics['throughput']['total_tokens']}")
    print(f"Total requests: {metrics['throughput']['total_requests']}")

    print("="*60)

# Run benchmark
if __name__ == "__main__":
    benchmark_inference(num_runs=10)
```

**Expected Output on Tesla T4:**
```
Warming up...
Running 10 iterations...

Run 1/10: 134.2 tok/s, 690ms
Run 2/10: 136.5 tok/s, 685ms
Run 3/10: 133.8 tok/s, 695ms
[...]

============================================================
📊 Benchmark Results
============================================================

Throughput (tokens/sec):
  Mean:   134.52
  Median: 134.30
  Stdev:  1.24
  Min:    132.80
  Max:    136.50

Latency (ms):
  Mean:   692.45
  Median: 690.00
  Stdev:  8.32
  Min:    685.00
  Max:    710.00

Latency Percentiles:
  P50:    690.00ms
  P95:    705.00ms
  P99:    710.00ms

Total tokens generated: 940
Total requests: 10
============================================================
```

---

## Advanced: Custom Chat Engine

Using the ChatEngine for conversations.

```python
from llcuda import InferenceEngine
from llcuda.chat import ChatEngine

def advanced_chat_example():
    """Advanced chat with conversation history."""

    # Initialize inference engine
    engine = InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )

    # Create chat engine with system prompt
    chat = ChatEngine(
        engine,
        system_prompt="You are a helpful AI coding assistant specialized in Python.",
        max_history=20,
        max_tokens=200,
        temperature=0.7
    )

    # Conversation
    chat.add_user_message("How do I read a file in Python?")
    response1 = chat.complete()
    print(f"AI: {response1}\n")

    chat.add_user_message("Can you show me an example?")
    response2 = chat.complete()
    print(f"AI: {response2}\n")

    chat.add_user_message("What about writing to a file?")
    response3 = chat.complete()
    print(f"AI: {response3}\n")

    # Get conversation history
    history = chat.get_history()
    print(f"Conversation has {len(history)} messages")

    # Save conversation
    chat.save_history("conversation.json")
    print("✅ Conversation saved to conversation.json")

    # Token count
    token_count = chat.count_tokens()
    print(f"Approximate token count: {token_count}")

# Run advanced chat
if __name__ == "__main__":
    advanced_chat_example()
```

---

## See Also

- [API Overview](overview.md) - Complete API reference
- [InferenceEngine](inference-engine.md) - Detailed engine documentation
- [Quick Start](../guides/quickstart.md) - Getting started guide
- [Tutorials](../tutorials/gemma-3-1b-colab.md) - Step-by-step tutorials
- [Performance](../performance/benchmarks.md) - Benchmark results
