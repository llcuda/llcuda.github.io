# Setup Short URL: llcuda.github.io

**Goal**: Change from `https://waqasm86.github.io/llcuda.github.io/` to `https://llcuda.github.io/`

---

## Steps to Get Short URL

### Step 1: Create GitHub Organization "llcuda"

You need to create a GitHub organization named **llcuda**.

**Instructions**:

1. Go to: https://github.com/settings/organizations
2. Click **"New organization"**
3. Choose a plan:
   - **Free** (recommended for open source projects)
   - Or any paid plan if needed
4. Fill in organization details:
   - **Organization account name**: `llcuda`
   - **Contact email**: Your email
   - **This organization belongs to**: Choose "My personal account"
5. Click **"Next"**
6. Add members (optional - you can skip this)
7. Click **"Complete setup"**

**Result**: Organization created at `https://github.com/llcuda`

---

### Step 2: Transfer Repository to Organization

Once the organization is created, transfer the repository:

**Option A: Via GitHub Web Interface**

1. Go to: https://github.com/waqasm86/llcuda.github.io/settings
2. Scroll down to **"Danger Zone"**
3. Click **"Transfer"**
4. Enter the new owner: `llcuda`
5. Enter repository name to confirm: `llcuda.github.io`
6. Click **"I understand, transfer this repository"**

**Option B: Via CLI (after organization is created)**

```bash
# You need to run this after creating the organization
gh repo transfer waqasm86/llcuda.github.io llcuda --yes
```

---

### Step 3: Verify GitHub Pages Settings

After transfer, verify GitHub Pages is configured:

1. Go to: https://github.com/llcuda/llcuda.github.io/settings/pages
2. Verify settings:
   - **Source**: Deploy from a branch
   - **Branch**: `gh-pages` / `root`
3. GitHub Pages will rebuild automatically

**Result**: Website will be available at `https://llcuda.github.io/`

---

### Step 4: Update Local Repository Remote

After transferring, update your local repository:

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/llcuda.github.io

# Update remote URL
git remote set-url origin https://github.com/llcuda/llcuda.github.io.git

# Verify new remote
git remote -v

# Pull to sync
git pull origin main
```

---

### Step 5: Update mkdocs.yml

Update the site_url in mkdocs.yml:

```bash
# Edit mkdocs.yml
nano mkdocs.yml
```

Change:
```yaml
site_url: https://waqasm86.github.io/llcuda.github.io/
```

To:
```yaml
site_url: https://llcuda.github.io/
```

Also update all URL references in:
- `docs/index.md`
- `docs/llms.txt`
- `docs/manifest.json`
- `docs/javascripts/schema.js`
- `SEO_IMPROVEMENTS.md`

---

### Step 6: Commit and Redeploy

```bash
# Commit changes
git add .
git commit -m "Update URLs for llcuda organization

- Changed site_url to https://llcuda.github.io/
- Updated all URL references in documentation
- Updated SEO files and meta tags
"

# Push to new remote
git push origin main

# Redeploy to GitHub Pages
mkdocs gh-deploy
```

---

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| **Organization** | waqasm86 (user) | llcuda (org) |
| **Repository** | waqasm86/llcuda.github.io | llcuda/llcuda.github.io |
| **Website URL** | https://waqasm86.github.io/llcuda.github.io/ | https://llcuda.github.io/ |
| **Git Remote** | git@github.com:waqasm86/llcuda.github.io.git | git@github.com:llcuda/llcuda.github.io.git |

---

## Important Notes

### Organization Benefits

✅ **Shorter URL**: `https://llcuda.github.io/`
✅ **Professional branding**: Organization looks more official
✅ **Team collaboration**: Can add multiple collaborators
✅ **Separate identity**: Project has its own identity

### Things to Update After Transfer

1. **Main llcuda repository**: Update documentation links
2. **Colab notebooks**: Update documentation URLs
3. **README files**: Update links to documentation
4. **Social media**: Update shared links

### GitHub Pages Limits

- **Free organizations**: GitHub Pages is free for public repositories
- **Custom domain**: You can optionally set up `docs.llcuda.com` or similar
- **HTTPS**: Automatically enabled by GitHub

---

## Quick Command Reference

After organization is created and repository is transferred:

```bash
# Navigate to project
cd /media/waqasm86/External1/Project-Nvidia-Office/llcuda.github.io

# Update remote URL
git remote set-url origin https://github.com/llcuda/llcuda.github.io.git

# Verify
git remote -v

# Pull latest
git pull origin main

# The rest will be automated via script
```

---

## Need Help?

If you encounter issues:

1. **Organization creation issues**: Check https://github.com/settings/organizations
2. **Transfer issues**: Ensure you have admin access to the repository
3. **GitHub Pages issues**: Check https://github.com/llcuda/llcuda.github.io/settings/pages

---

## Alternative: Custom Domain (Optional)

If you own a domain (e.g., `llcuda.com`), you can set up:
- `https://llcuda.com/` or
- `https://docs.llcuda.com/`

This gives even more professional appearance without needing GitHub organization.

To set up custom domain:
1. Go to repository settings → Pages
2. Add custom domain
3. Update DNS records at your domain registrar
4. GitHub provides SSL certificate automatically

---

**Next Action**: Create the "llcuda" organization on GitHub, then proceed with Step 2.

**Created**: January 10, 2026
