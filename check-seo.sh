#!/bin/bash
# SEO Health Check for llcuda.github.io

echo "========================================"
echo "llcuda.github.io SEO Health Check"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check 1: robots.txt
echo "1. Checking robots.txt..."
if [ -f "robots.txt" ]; then
    echo -e "${GREEN}✓${NC} robots.txt exists"
    if grep -q "Sitemap:" robots.txt; then
        echo -e "${GREEN}✓${NC} Sitemap URL found in robots.txt"
    else
        echo -e "${RED}✗${NC} No sitemap URL in robots.txt"
    fi
else
    echo -e "${RED}✗${NC} robots.txt not found"
fi
echo ""

# Check 2: sitemap.xml
echo "2. Checking sitemap.xml..."
if [ -f "sitemap.xml" ]; then
    echo -e "${GREEN}✓${NC} sitemap.xml exists"

    # Count URLs
    url_count=$(grep -c "<url>" sitemap.xml)
    echo -e "${GREEN}✓${NC} Found $url_count URLs in sitemap"

    # Check last modified date
    if grep -q "2026-01-25" sitemap.xml; then
        echo -e "${GREEN}✓${NC} Sitemap updated to 2026-01-25"
    else
        echo -e "${RED}⚠${NC} Sitemap may have old dates"
    fi
else
    echo -e "${RED}✗${NC} sitemap.xml not found"
fi
echo ""

# Check 3: Structured Data
echo "3. Checking structured data..."
if [ -f "seo-schema.json" ]; then
    echo -e "${GREEN}✓${NC} seo-schema.json exists"

    if grep -q "\"version\": \"2.2.0\"" seo-schema.json; then
        echo -e "${GREEN}✓${NC} Version 2.2.0 in schema"
    else
        echo -e "${RED}⚠${NC} Version may be outdated in schema"
    fi
else
    echo -e "${RED}⚠${NC} seo-schema.json not found"
fi
echo ""

# Check 4: index.html meta tags
echo "4. Checking index.html meta tags..."
if [ -f "index.html" ]; then
    echo -e "${GREEN}✓${NC} index.html exists"

    if grep -q "v2.2.0" index.html; then
        echo -e "${GREEN}✓${NC} Version 2.2.0 found in index.html"
    else
        echo -e "${RED}⚠${NC} Version 2.2.0 not prominent in index.html"
    fi

    if grep -q "<meta name=\"description\"" index.html; then
        echo -e "${GREEN}✓${NC} Meta description tag present"
    else
        echo -e "${RED}✗${NC} No meta description tag"
    fi

    if grep -q "<title>" index.html; then
        echo -e "${GREEN}✓${NC} Title tag present"
        title=$(grep -o "<title>[^<]*</title>" index.html | head -1)
        echo "  Title: $title"
    else
        echo -e "${RED}✗${NC} No title tag"
    fi
else
    echo -e "${RED}✗${NC} index.html not found"
fi
echo ""

# Check 5: File sizes
echo "5. Checking file sizes..."
if [ -f "index.html" ]; then
    size=$(du -h index.html | cut -f1)
    echo "  index.html: $size"
fi

if [ -f "sitemap.xml" ]; then
    size=$(du -h sitemap.xml | cut -f1)
    echo "  sitemap.xml: $size"
fi
echo ""

# Check 6: URL structure
echo "6. Checking URL structure..."
if [ -d "guides" ]; then
    echo -e "${GREEN}✓${NC} /guides/ directory exists"
fi

if [ -d "tutorials" ]; then
    echo -e "${GREEN}✓${NC} /tutorials/ directory exists"
fi

if [ -d "api" ]; then
    echo -e "${GREEN}✓${NC} /api/ directory exists"
fi
echo ""

# Check 7: Git status
echo "7. Checking git status..."
if git status &>/dev/null; then
    untracked=$(git status --porcelain | grep "^??" | wc -l)
    modified=$(git status --porcelain | grep "^ M" | wc -l)

    if [ "$untracked" -gt 0 ]; then
        echo -e "${RED}⚠${NC} $untracked untracked files"
    fi

    if [ "$modified" -gt 0 ]; then
        echo -e "${RED}⚠${NC} $modified modified files"
    fi

    if [ "$untracked" -eq 0 ] && [ "$modified" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Working directory clean"
    fi
else
    echo -e "${RED}⚠${NC} Not a git repository"
fi
echo ""

# Summary
echo "========================================"
echo "SEO Health Check Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Add SEO optimization for v2.2.0'"
echo "3. git push origin gh-pages"
echo "4. Submit sitemap to Google Search Console"
echo "5. Request indexing for key pages"
echo ""
echo "Documentation: SEO_OPTIMIZATION.md"
