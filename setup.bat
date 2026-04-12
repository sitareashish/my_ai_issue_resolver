@echo off
echo ============================================
echo   GitHub Issue Resolver - Windows Setup
echo ============================================
echo.

REM Step 1: Rename files that need a dot prefix
echo [1/4] Renaming config files...
if exist "env.example"              rename "env.example"              ".env.example"
if exist "gitignore.txt"            rename "gitignore.txt"            ".gitignore"
if exist "pre-commit-config.yaml"   rename "pre-commit-config.yaml"   ".pre-commit-config.yaml"

REM Step 2: Create .github/workflows folder and move CI file
echo [2/4] Setting up GitHub Actions folder...
if not exist ".github\workflows" mkdir ".github\workflows"
if exist "github_workflows\ci.yml" (
    copy "github_workflows\ci.yml" ".github\workflows\ci.yml"
    rmdir /s /q "github_workflows"
)

REM Step 3: Create .env from example
echo [3/4] Creating .env file...
if not exist ".env" (
    copy ".env.example" ".env"
    echo.
    echo  IMPORTANT: Open .env and fill in your API keys before running!
    echo  Keys needed:
    echo    ANTHROPIC_API_KEY
    echo    OPENAI_API_KEY
    echo    GITHUB_TOKEN
    echo    GITHUB_REPO
)

REM Step 4: Git init
echo [4/4] Initialising git repository...
git init
git add .
git commit -m "feat: initial commit - GitHub Issue Resolver"

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Edit .env with your API keys
echo   2. Create a repo on github.com/new
echo   3. Run these commands:
echo      git remote add origin https://github.com/YOUR-USERNAME/github-issue-resolver.git
echo      git branch -M main
echo      git push -u origin main
echo.
echo Then to run:
echo   pip install -e .[dev]
echo   make docker-up
echo   make run ISSUE=42
echo.
pause
