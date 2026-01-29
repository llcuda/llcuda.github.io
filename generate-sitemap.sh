#!/bin/bash
echo '<?xml version="1.0" encoding="UTF-8"?>'
echo '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'

# Homepage - highest priority
cat << XML
    <url>
        <loc>https://llcuda.github.io/</loc>
        <lastmod>2026-01-25</lastmod>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
    </url>
XML

# Tutorials - high priority
cat << XML
    <url>
        <loc>https://llcuda.github.io/tutorials/</loc>
        <lastmod>2026-01-25</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.9</priority>
    </url>
    <url>
        <loc>https://llcuda.github.io/tutorials/01-quickstart/</loc>
        <lastmod>2026-01-25</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.8</priority>
    </url>
XML

# Continue for all pages...

echo '</urlset>'
