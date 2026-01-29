#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

# Convert file paths to URLs
file_list = [
    "./api/client/index.html",
    "./api/examples/index.html",
    "./api/gguf/index.html",
    "./api/graphistry/index.html",
    "./api/models/index.html",
    "./api/multigpu/index.html",
    "./api/nccl/index.html",
    "./api/overview/index.html",
    "./api/server/index.html",
    "./architecture/gpu0-llm/index.html",
    "./architecture/gpu1-graphistry/index.html",
    "./architecture/overview/index.html",
    "./architecture/split-gpu/index.html",
    "./architecture/tensor-split-vs-nccl/index.html",
    "./gguf/i-quants/index.html",
    "./gguf/k-quants/index.html",
    "./gguf/overview/index.html",
    "./gguf/selection/index.html",
    "./gguf/vram-estimation/index.html",
    "./graphistry/examples/index.html",
    "./graphistry/knowledge-graphs/index.html",
    "./graphistry/overview/index.html",
    "./graphistry/rapids/index.html",
    "./graphistry/split-gpu-setup/index.html",
    "./guides/build-from-source/index.html",
    "./guides/faq/index.html",
    "./guides/first-steps/index.html",
    "./guides/installation/index.html",
    "./guides/kaggle-setup/index.html",
    "./guides/model-selection/index.html",
    "./guides/quickstart/index.html",
    "./guides/troubleshooting/index.html",
    "./index.html",
    "./kaggle/dual-gpu-setup/index.html",
    "./kaggle/large-models/index.html",
    "./kaggle/multi-gpu-inference/index.html",
    "./kaggle/overview/index.html",
    "./kaggle/tensor-split/index.html",
    "./performance/benchmarks/index.html",
    "./performance/dual-t4-results/index.html",
    "./performance/flash-attention/index.html",
    "./performance/memory/index.html",
    "./performance/optimization/index.html",
    "./tutorials/01-quickstart/index.html",
    "./tutorials/02-server-setup/index.html",
    "./tutorials/03-multi-gpu/index.html",
    "./tutorials/04-gguf-quantization/index.html",
    "./tutorials/05-unsloth-integration/index.html",
    "./tutorials/06-split-gpu-graphistry/index.html",
    "./tutorials/07-openai-api/index.html",
    "./tutorials/08-nccl-pytorch/index.html",
    "./tutorials/09-large-models/index.html",
    "./tutorials/10-complete-workflow/index.html",
    "./tutorials/11-gguf-neural-network-visualization/index.html",
    "./tutorials/index.html",
    "./unsloth/best-practices/index.html",
    "./unsloth/deployment/index.html",
    "./unsloth/fine-tuning/index.html",
    "./unsloth/gguf-export/index.html",
    "./unsloth/overview/index.html"
]

# Function to convert file path to URL
def path_to_url(path):
    # Remove leading ./
    if path.startswith("./"):
        path = path[2:]
    # Remove index.html
    if path.endswith("index.html"):
        path = path[:-10]  # Remove "index.html"
    elif path.endswith("/index.html"):
        path = path[:-11]  # Remove "/index.html"
    # Ensure no trailing slash for homepage
    if path == "":
        return "https://llcuda.github.io/"
    else:
        return f"https://llcuda.github.io/{path.rstrip('/')}/"

# Create XML
urlset = ET.Element("urlset")
urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

today = datetime.now().strftime("%Y-%m-%d")

# First add homepage with highest priority
homepage = ET.SubElement(urlset, "url")
ET.SubElement(homepage, "loc").text = "https://llcuda.github.io/"
ET.SubElement(homepage, "lastmod").text = today
ET.SubElement(homepage, "changefreq").text = "daily"
ET.SubElement(homepage, "priority").text = "1.0"

# Add all other pages
for file_path in file_list:
    if file_path == "./index.html":
        continue  # Already added homepage
    
    url = path_to_url(file_path)
    
    url_elem = ET.SubElement(urlset, "url")
    ET.SubElement(url_elem, "loc").text = url
    ET.SubElement(url_elem, "lastmod").text = today
    
    # Set changefreq based on section
    if "tutorial" in url or "guide" in url or "quickstart" in url:
        ET.SubElement(url_elem, "changefreq").text = "monthly"
        ET.SubElement(url_elem, "priority").text = "0.9"
    elif "api" in url or "architecture" in url:
        ET.SubElement(url_elem, "changefreq").text = "monthly"
        ET.SubElement(url_elem, "priority").text = "0.8"
    elif "tutorials/11" in url:  # Special tutorial
        ET.SubElement(url_elem, "changefreq").text = "weekly"
        ET.SubElement(url_elem, "priority").text = "0.9"
    else:
        ET.SubElement(url_elem, "changefreq").text = "monthly"
        ET.SubElement(url_elem, "priority").text = "0.7"

# Pretty print
xml_str = ET.tostring(urlset, encoding="unicode")
dom = minidom.parseString(xml_str)
pretty_xml = dom.toprettyxml(indent="    ")

# Remove XML declaration from minidom and add our own
lines = pretty_xml.split('\n')
xml_content = '\n'.join(lines[1:])  # Skip first line

with open("sitemap.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write(xml_content)

print(f"✅ Generated sitemap.xml with {len(file_list)} pages")
print(f"📊 Total URLs in sitemap: {len(file_list)}")
