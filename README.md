# llcuda.github.io

Official documentation website for llcuda v2.2.0.

## Development

```bash
# Install dependencies
pip install mkdocs-material mkdocs-minify-plugin

# Serve locally
mkdocs serve

# View at http://127.0.0.1:8000
```

## Deployment

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Structure

- `mkdocs.yml` - Site configuration
- `docs/` - Documentation source
  - `index.md` - Homepage
  - `guides/` - Getting started guides
  - `kaggle/` - Kaggle dual T4 guides
  - `tutorials/` - Tutorial notebooks
  - `api/` - API reference
  - `architecture/` - System architecture
  - `unsloth/` - Unsloth integration
  - `graphistry/` - Graphistry visualization
  - `performance/` - Benchmarks & optimization
  - `gguf/` - GGUF quantization

## Version

llcuda v2.2.0 - CUDA12 Inference Backend for Unsloth
