// Schema.org structured data for SEO
(function() {
  const schema = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "llcuda v2.2.0",
    "applicationCategory": "DeveloperApplication",
    "applicationSubCategory": "Machine Learning Framework",
    "operatingSystem": "Linux",
    "offers": {
      "@type": "Offer",
      "price": "0",
      "priceCurrency": "USD"
    },
    "description": "CUDA 12 inference backend for Unsloth optimized for small GGUF models (1B-5B) on Kaggle dual Tesla T4 GPUs (15GB × 2, SM 7.5). Split-GPU architecture with LLM inference on GPU 0 and Graphistry neural network visualization on GPU 1. Features 11 comprehensive tutorials including groundbreaking GGUF architecture visualization with 929 nodes, 981 edges, and 896 attention heads. Built-in llama.cpp server and NVIDIA NCCL.",
    "softwareVersion": "2.2.0",
    "author": {
      "@type": "Person",
      "name": "Waqas Muhammad",
      "email": "waqasm86@gmail.com",
      "url": "https://github.com/waqasm86"
    },
    "creator": {
      "@type": "Person",
      "name": "Waqas Muhammad",
      "email": "waqasm86@gmail.com",
      "url": "https://github.com/waqasm86"
    },
    "datePublished": "2026-01-10",
    "dateModified": "2026-01-22",
    "url": "https://llcuda.github.io/",
    "image": "https://llcuda.github.io/assets/images/social-card.png",
    "screenshot": "https://llcuda.github.io/assets/images/screenshot.png",
    "codeRepository": "https://github.com/llcuda/llcuda",
    "downloadUrl": "https://github.com/llcuda/llcuda/releases",
    "installUrl": "https://llcuda.github.io/guides/installation/",
    "softwareRequirements": "Python 3.11+, CUDA 12.x, Kaggle Dual Tesla T4 GPUs",
    "memoryRequirements": "30GB VRAM (15GB × 2 Tesla T4)",
    "processorRequirements": "NVIDIA Tesla T4 (SM 7.5) or compatible GPU",
    "programmingLanguage": [
      "Python",
      "C++",
      "CUDA"
    ],
    "keywords": "llcuda, llcuda v2.2.0, CUDA 12, Tesla T4, Kaggle, dual GPU, LLM inference, Unsloth, GGUF quantization, 1B-5B models, small models, llama.cpp, multi-GPU, tensor-split, Graphistry, neural network visualization, GGUF visualization, 929 nodes, 981 edges, 896 attention heads, FlashAttention, split-GPU architecture, RAPIDS cuGraph, PageRank, transformer architecture, Llama-3.2-3B, interactive dashboards, NVIDIA NCCL",
    "license": "https://opensource.org/licenses/MIT",
    "aggregateRating": {
      "@type": "AggregateRating",
      "ratingValue": "5",
      "ratingCount": "1"
    },
    "featureList": [
      "Optimized for 1B-5B parameter models on Kaggle dual T4",
      "Split-GPU architecture: LLM on GPU 0, Graphistry on GPU 1",
      "11 comprehensive tutorials from beginner to advanced",
      "Tutorial 11: GGUF neural network visualization (929 nodes, 981 edges)",
      "896 attention heads visualization across 28 transformer layers",
      "Interactive Graphistry dashboards with cloud URLs",
      "GPU-accelerated PageRank and centrality analytics",
      "Built-in llama.cpp server with OpenAI-compatible API",
      "29 GGUF quantization formats (K-quants, I-quants)",
      "FlashAttention v2 support (2-3x faster inference)",
      "Tensor Core optimization (SM 7.5)",
      "NVIDIA NCCL for PyTorch distributed training",
      "Unsloth integration for fine-tuning workflow",
      "Auto-download CUDA 12.5 binaries from GitHub Releases",
      "Knowledge graph extraction and visualization",
      "RAPIDS cuDF and cuGraph integration"
    ]
  };

  // Create and append script tag
  const script = document.createElement('script');
  script.type = 'application/ld+json';
  script.text = JSON.stringify(schema, null, 2);
  document.head.appendChild(script);

  // Add breadcrumb schema
  const breadcrumbSchema = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    "itemListElement": [{
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "https://llcuda.github.io/"
    }]
  };

  const breadcrumbScript = document.createElement('script');
  breadcrumbScript.type = 'application/ld+json';
  breadcrumbScript.text = JSON.stringify(breadcrumbSchema, null, 2);
  document.head.appendChild(breadcrumbScript);

  // Add WebSite schema
  const websiteSchema = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "llcuda v2.2.0 Documentation - CUDA12 Inference for 1B-5B Models on Kaggle",
    "alternateName": "llcuda Documentation",
    "url": "https://llcuda.github.io/",
    "description": "Official documentation for llcuda v2.2.0 - CUDA 12 inference backend optimized for small GGUF models (1B-5B) on Kaggle dual Tesla T4 GPUs with split-GPU architecture and neural network visualization. Features 11 comprehensive tutorials including GGUF architecture visualization with 929 nodes and 981 edges.",
    "inLanguage": "en",
    "potentialAction": {
      "@type": "SearchAction",
      "target": "https://llcuda.github.io/search/?q={search_term_string}",
      "query-input": "required name=search_term_string"
    },
    "about": {
      "@type": "SoftwareApplication",
      "name": "llcuda v2.2.0",
      "description": "CUDA 12 inference backend for Unsloth with neural network visualization"
    }
  };

  const websiteScript = document.createElement('script');
  websiteScript.type = 'application/ld+json';
  websiteScript.text = JSON.stringify(websiteSchema, null, 2);
  document.head.appendChild(websiteScript);

  // Add TechArticle schema for documentation
  const articleSchema = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    "headline": "llcuda v2.2.0 - CUDA12 Inference Backend for 1B-5B Models on Kaggle Dual T4 with Neural Network Visualization",
    "description": "Comprehensive documentation for llcuda v2.2.0, a CUDA 12 inference backend optimized for small GGUF models (1B-5B) on Kaggle dual Tesla T4 GPUs. Features split-GPU architecture with LLM inference and Graphistry neural network visualization. Includes 11 tutorials with groundbreaking GGUF architecture visualization (929 nodes, 981 edges, 896 attention heads).",
    "author": {
      "@type": "Person",
      "name": "Waqas Muhammad",
      "email": "waqasm86@gmail.com",
      "url": "https://github.com/waqasm86"
    },
    "datePublished": "2026-01-10",
    "dateModified": "2026-01-22",
    "publisher": {
      "@type": "Person",
      "name": "Waqas Muhammad"
    },
    "mainEntityOfPage": {
      "@type": "WebPage",
      "@id": "https://llcuda.github.io/"
    },
    "about": [
      {
        "@type": "Thing",
        "name": "CUDA 12 Inference"
      },
      {
        "@type": "Thing",
        "name": "Neural Network Visualization"
      },
      {
        "@type": "Thing",
        "name": "GGUF Quantization"
      },
      {
        "@type": "Thing",
        "name": "Kaggle Dual T4 GPUs"
      },
      {
        "@type": "Thing",
        "name": "Split-GPU Architecture"
      }
    ]
  };

  const articleScript = document.createElement('script');
  articleScript.type = 'application/ld+json';
  articleScript.text = JSON.stringify(articleSchema, null, 2);
  document.head.appendChild(articleScript);
})();
