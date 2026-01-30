import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from datetime import datetime

urls = [
    # Homepage
    ("https://llcuda.github.io/", "daily", "1.0"),
    
    # Guides
    ("https://llcuda.github.io/guides/quickstart/", "monthly", "0.9"),
    ("https://llcuda.github.io/guides/installation/", "monthly", "0.9"),
    ("https://llcuda.github.io/guides/first-steps/", "monthly", "0.9"),
    ("https://llcuda.github.io/guides/kaggle-setup/", "monthly", "0.9"),
    
    # Tutorials (11 of them)
    ("https://llcuda.github.io/tutorials/", "weekly", "0.9"),
    ("https://llcuda.github.io/tutorials/01-quickstart/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/02-server-setup/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/03-multi-gpu/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/04-gguf-quantization/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/05-unsloth-integration/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/06-split-gpu-graphistry/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/07-openai-api/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/08-nccl-pytorch/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/09-large-models/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/10-complete-workflow/", "monthly", "0.8"),
    ("https://llcuda.github.io/tutorials/11-gguf-neural-network-visualization/", "weekly", "0.9"),
    
    # API
    ("https://llcuda.github.io/api/overview/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/inference-engine/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/device/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/client/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/server/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/examples/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/gguf/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/graphistry/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/models/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/multigpu/", "monthly", "0.8"),
    ("https://llcuda.github.io/api/nccl/", "monthly", "0.8"),
    
    # Add more URLs from your file list...
]

# Create XML
urlset = ET.Element("urlset")
urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

today = datetime.now().strftime("%Y-%m-%d")

for loc, changefreq, priority in urls:
    url_elem = ET.SubElement(urlset, "url")
    ET.SubElement(url_elem, "loc").text = loc
    ET.SubElement(url_elem, "lastmod").text = today
    ET.SubElement(url_elem, "changefreq").text = changefreq
    ET.SubElement(url_elem, "priority").text = priority

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

print(f"Generated sitemap with {len(urls)} URLs")
