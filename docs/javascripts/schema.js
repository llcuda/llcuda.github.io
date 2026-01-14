// Schema.org structured data for SEO
(function() {
  const schema = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "llcuda",
    "applicationCategory": "DeveloperApplication",
    "operatingSystem": "Linux",
    "offers": {
      "@type": "Offer",
      "price": "0",
      "priceCurrency": "USD"
    },
    "description": "Fast LLM inference on Tesla T4 GPUs with CUDA 12, FlashAttention, and Unsloth integration. Verified 134 tokens/sec on Gemma 3-1B.",
    "softwareVersion": "2.1.0",
    "author": {
      "@type": "Person",
      "name": "Waqas Muhammad",
      "url": "https://github.com/waqasm86"
    },
    "creator": {
      "@type": "Person",
      "name": "Waqas Muhammad",
      "url": "https://github.com/waqasm86"
    },
    "datePublished": "2026-01-10",
    "dateModified": "2026-01-10",
    "url": "https://llcuda.github.io/",
    "image": "https://llcuda.github.io/assets/images/social-card.png",
    "screenshot": "https://llcuda.github.io/assets/images/screenshot.png",
    "codeRepository": "https://github.com/llcuda/llcuda",
    "programmingLanguage": [
      "Python",
      "C++",
      "CUDA"
    ],
    "keywords": "llcuda, CUDA, Tesla T4, LLM inference, FlashAttention, GGUF, Unsloth, Google Colab, GPU acceleration",
    "license": "https://opensource.org/licenses/MIT",
    "aggregateRating": {
      "@type": "AggregateRating",
      "ratingValue": "5",
      "ratingCount": "1"
    },
    "featureList": [
      "134 tokens/sec verified performance on Tesla T4",
      "FlashAttention v2 support (2-3x faster)",
      "Tensor Core optimization (SM 7.5)",
      "GitHub-only distribution with auto-download",
      "GGUF format support",
      "Google Colab optimized",
      "Unsloth integration"
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
    "name": "llcuda Documentation",
    "url": "https://llcuda.github.io/",
    "description": "Official documentation for llcuda v2.1.0 - Tesla T4 CUDA Inference Engine",
    "potentialAction": {
      "@type": "SearchAction",
      "target": "https://llcuda.github.io/search/?q={search_term_string}",
      "query-input": "required name=search_term_string"
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
    "headline": "llcuda v2.1.0 - Tesla T4 CUDA Inference Documentation",
    "description": "Comprehensive documentation for llcuda, a fast LLM inference engine for Tesla T4 GPUs",
    "author": {
      "@type": "Person",
      "name": "Waqas Muhammad",
      "url": "https://github.com/waqasm86"
    },
    "datePublished": "2026-01-10",
    "dateModified": "2026-01-10",
    "publisher": {
      "@type": "Person",
      "name": "Waqas Muhammad"
    },
    "mainEntityOfPage": {
      "@type": "WebPage",
      "@id": "https://llcuda.github.io/"
    }
  };

  const articleScript = document.createElement('script');
  articleScript.type = 'application/ld+json';
  articleScript.text = JSON.stringify(articleSchema, null, 2);
  document.head.appendChild(articleScript);
})();
