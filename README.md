# llcuda Documentation Website

Official documentation for **llcuda v2.0.6** - Tesla T4 CUDA Inference

This repository contains the source files for the llcuda documentation website built with MkDocs Material.

## 🌐 Live Website

**Coming soon**: https://llcuda.github.io/

## 📚 Documentation Structure

```
docs/
├── index.md                    # Homepage
├── guides/
│   ├── quickstart.md          # 5-minute quick start
│   ├── installation.md        # Complete installation guide
│   ├── first-steps.md         # First steps after installation
│   ├── model-selection.md     # Choosing the right model
│   ├── gguf-format.md         # Understanding GGUF format
│   ├── troubleshooting.md     # Common issues and solutions
│   └── faq.md                 # Frequently asked questions
├── tutorials/
│   ├── gemma-3-1b-colab.md    # Gemma 3-1B Google Colab tutorial
│   ├── gemma-3-1b-executed.md # Live execution output
│   ├── build-binaries.md      # Build CUDA binaries
│   ├── unsloth-integration.md # Unsloth workflow
│   └── performance.md         # Performance optimization
├── api/
│   ├── overview.md            # API overview
│   ├── inference-engine.md    # InferenceEngine class
│   ├── models.md              # Models and GGUF
│   ├── device.md              # GPU and device management
│   └── examples.md            # Code examples
├── performance/
│   ├── benchmarks.md          # Performance benchmarks
│   ├── t4-results.md          # Tesla T4 detailed results
│   └── optimization.md        # Optimization guide
└── notebooks/
    ├── index.md               # Notebooks overview
    └── colab.md               # Colab notebooks guide
```

## 🚀 Local Development

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Clone this repository
git clone https://github.com/waqasm86/llcuda.github.io.git
cd llcuda.github.io

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Start development server
mkdocs serve

# Open in browser
# http://127.0.0.1:8000/
```

The site will auto-reload when you make changes to the documentation.

### Build Static Site

```bash
# Build the documentation
mkdocs build

# Output will be in site/ directory
```

## 📝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test locally with `mkdocs serve`
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

### Writing Guidelines

- Use clear, concise language
- Include code examples where appropriate
- Add Google Colab badges for notebook links
- Follow the existing structure and formatting
- Test all code examples before submitting

## 🎨 Theme

This documentation uses [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with custom configuration for:

- Dark/light mode toggle
- Syntax highlighting
- Search functionality
- Social links
- Google Analytics integration
- Cookie consent

## 📦 Main Project

This is the documentation website for llcuda. The main project repository is:

**https://github.com/waqasm86/llcuda**

## 📄 License

This documentation is licensed under MIT License, same as the main llcuda project.

Copyright © 2024-2026 Waqas Muhammad

---

**Built with**: [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) | **Hosted on**: GitHub Pages
