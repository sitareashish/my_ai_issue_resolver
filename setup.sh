#!/bin/bash
echo "============================================"
echo "  GitHub Issue Resolver - Setup"
echo "============================================"
echo

# Step 1: Rename files that need a dot prefix
echo "[1/4] Renaming config files..."
[ -f "env.example" ]            && mv "env.example"            ".env.example"
[ -f "gitignore.txt" ]          && mv "gitignore.txt"          ".gitignore"
[ -f "pre-commit-config.yaml" ] && mv "pre-commit-config.yaml" ".pre-commit-config.yaml"

# Step 2: Create .github/workflows and move CI file
echo "[2/4] Setting up GitHub Actions folder..."
mkdir -p .github/workflows
if [ -f "github_workflows/ci.yml" ]; then
    mv github_workflows/ci.yml .github/workflows/ci.yml
    rm -rf github_workflows
fi

# Step 3: Create .env from example
echo "[3/4] Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    echo "  ⚠️  Open .env and fill in your API keys before running!"
fi

# Step 4: Git init
echo "[4/4] Initialising git repository..."
git init
git add .
git commit -m "feat: initial commit - GitHub Issue Resolver"

echo
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Create a repo at github.com/new"
echo "  3. Run:"
echo "     git remote add origin https://github.com/YOUR-USERNAME/github-issue-resolver.git"
echo "     git branch -M main"
echo "     git push -u origin main"
echo
echo "Then to run the pipeline:"
echo "  pip install -e '.[dev]'"
echo "  make docker-up"
echo "  make run ISSUE=42"
