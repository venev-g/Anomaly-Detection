#!/bin/bash

# Automated Git Commit and Sync Script
# Commits each file individually with meaningful commit messages
# Excludes files larger than 90MB from GitHub upload

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_FILE_SIZE_MB=90
REPO_DIR="/workspaces/Anomaly-Detection"
LARGE_FILES_LIST="large_files_excluded.txt"

echo -e "${BLUE}🚀 Starting Automated Git Commit and Sync Process${NC}"
echo "=================================================="

# Navigate to repository directory
cd "$REPO_DIR"

# Function to get human readable file size
get_file_size_mb() {
    local file="$1"
    if [[ -f "$file" ]]; then
        local size_bytes=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
        echo $((size_bytes / 1024 / 1024))
    else
        echo "0"
    fi
}

# Function to generate meaningful commit message based on file type and content
generate_commit_message() {
    local file="$1"
    local action="$2"  # "add", "modify", "delete"
    local filename=$(basename "$file")
    local dirname=$(dirname "$file")
    local extension="${filename##*.}"
    
    # Generate message based on file type and location
    case "$file" in
        # Documentation files
        "README.md"|"*/README.md")
            echo "docs: ${action} project README with setup and usage instructions"
            ;;
        "GETTING_STARTED.md"|"*/GETTING_STARTED.md")
            echo "docs: ${action} comprehensive getting started guide with step-by-step instructions"
            ;;
        "QUICKSTART.md"|"*/QUICKSTART.md")
            echo "docs: ${action} quick start guide for rapid deployment"
            ;;
        
        # Configuration files
        "config.yaml"|"*/config.yaml")
            echo "config: ${action} system configuration with XGBoost and ZenML settings"
            ;;
        "requirements.txt"|"*/requirements.txt")
            echo "deps: ${action} Python dependencies for anomaly detection system"
            ;;
        
        # Core source code
        "src/"*".py")
            local module_name=$(basename "$file" .py)
            case "$module_name" in
                "data_ingester")
                    echo "feat: ${action} KDD99 data ingestion module with strategy pattern"
                    ;;
                "data_preprocessor")
                    echo "feat: ${action} data preprocessing pipeline with feature engineering"
                    ;;
                "model_builder")
                    echo "feat: ${action} XGBoost model builder with GPU acceleration support"
                    ;;
                "model_evaluator")
                    echo "feat: ${action} model evaluation framework with comprehensive metrics"
                    ;;
                *)
                    echo "feat: ${action} ${module_name} module for anomaly detection system"
                    ;;
            esac
            ;;
        
        # ZenML pipeline steps
        "steps/"*".py")
            local step_name=$(basename "$file" .py)
            echo "pipeline: ${action} ZenML pipeline step - ${step_name}"
            ;;
        
        # Pipeline definitions
        "pipelines/"*".py")
            local pipeline_name=$(basename "$file" .py)
            case "$pipeline_name" in
                "training_pipeline")
                    echo "pipeline: ${action} training pipeline for model development and evaluation"
                    ;;
                "deployment_pipeline")
                    echo "pipeline: ${action} deployment pipeline for model serving with MLflow"
                    ;;
                *)
                    echo "pipeline: ${action} ${pipeline_name} pipeline definition"
                    ;;
            esac
            ;;
        
        # Analysis and EDA
        "analysis/"*".py")
            echo "analysis: ${action} EDA analysis module for data insights"
            ;;
        "analysis/"*".ipynb")
            echo "analysis: ${action} comprehensive EDA Jupyter notebook for KDD99 dataset"
            ;;
        
        # Explanation and examples
        "explanations/"*".py")
            local pattern_name=$(basename "$file" .py)
            echo "docs: ${action} design pattern example - ${pattern_name}"
            ;;
        
        # Main execution scripts
        "run_pipeline.py")
            echo "feat: ${action} main training pipeline executor with configuration support"
            ;;
        "run_deployment.py")
            echo "feat: ${action} model deployment script with MLflow integration"
            ;;
        "sample_predict.py")
            echo "feat: ${action} sample prediction script for testing deployed models"
            ;;
        "verify_integration.py")
            echo "test: ${action} ZenML/MLflow integration verification script"
            ;;
        
        # VS Code settings
        ".vscode/"*)
            echo "dev: ${action} VS Code workspace configuration and settings"
            ;;
        
        # Data files (should be excluded if large)
        "data/"*|"*/data/"*)
            echo "data: ${action} dataset file for anomaly detection (WARNING: Check file size)"
            ;;
        
        # Default cases by extension
        *.py)
            echo "feat: ${action} Python module - ${filename}"
            ;;
        *.yaml|*.yml)
            echo "config: ${action} configuration file - ${filename}"
            ;;
        *.md)
            echo "docs: ${action} documentation file - ${filename}"
            ;;
        *.txt)
            echo "docs: ${action} text file - ${filename}"
            ;;
        *.json)
            echo "config: ${action} JSON configuration - ${filename}"
            ;;
        *)
            echo "misc: ${action} ${filename}"
            ;;
    esac
}

# Function to check if file should be excluded due to size
should_exclude_file() {
    local file="$1"
    local size_mb=$(get_file_size_mb "$file")
    
    if [[ $size_mb -gt $MAX_FILE_SIZE_MB ]]; then
        return 0  # Should exclude
    else
        return 1  # Should include
    fi
}

# Create/update .gitignore for large files
echo -e "${YELLOW}🔍 Identifying large files (>90MB) to exclude...${NC}"
> "$LARGE_FILES_LIST"

# Find and list large files
large_files_found=false
while IFS= read -r -d '' file; do
    if should_exclude_file "$file"; then
        echo "❌ Excluding large file: $file ($(get_file_size_mb "$file")MB)"
        echo "$file" >> "$LARGE_FILES_LIST"
        
        # Add to .gitignore if not already there
        if ! grep -Fxq "$file" .gitignore 2>/dev/null; then
            echo "$file" >> .gitignore
        fi
        large_files_found=true
    fi
done < <(find . -type f -print0)

if [[ "$large_files_found" == "true" ]]; then
    echo -e "${YELLOW}📝 Large files have been added to .gitignore${NC}"
    echo -e "${YELLOW}📋 List of excluded files saved to: $LARGE_FILES_LIST${NC}"
fi

# Get list of all changed files (excluding large ones)
echo -e "\n${BLUE}📋 Analyzing changed files...${NC}"

# Get modified files
modified_files=()
if git diff --name-only | grep -v "^$" > /dev/null 2>&1; then
    while IFS= read -r file; do
        if [[ -f "$file" ]] && ! should_exclude_file "$file"; then
            modified_files+=("$file")
        fi
    done < <(git diff --name-only)
fi

# Get untracked files
untracked_files=()
if git ls-files --others --exclude-standard | grep -v "^$" > /dev/null 2>&1; then
    while IFS= read -r file; do
        if [[ -f "$file" ]] && ! should_exclude_file "$file"; then
            untracked_files+=("$file")
        fi
    done < <(git ls-files --others --exclude-standard)
fi

# Combine all files
all_files=("${modified_files[@]}" "${untracked_files[@]}")

if [[ ${#all_files[@]} -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  No files to commit (all files may be too large or already committed)${NC}"
    exit 0
fi

echo -e "${GREEN}📂 Found ${#all_files[@]} files to commit${NC}"

# Commit each file individually
commit_count=0
failed_commits=0

for file in "${all_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${YELLOW}⚠️  Skipping non-existent file: $file${NC}"
        continue
    fi
    
    # Determine if this is a new file or modified file
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        action="update"
    else
        action="add"
    fi
    
    # Generate commit message
    commit_msg=$(generate_commit_message "$file" "$action")
    
    echo -e "\n${BLUE}📝 Committing: $file${NC}"
    echo -e "${BLUE}💬 Message: $commit_msg${NC}"
    
    # Add and commit the file
    if git add "$file" && git commit -m "$commit_msg"; then
        echo -e "${GREEN}✅ Successfully committed: $file${NC}"
        ((commit_count++))
    else
        echo -e "${RED}❌ Failed to commit: $file${NC}"
        ((failed_commits++))
    fi
done

# Summary
echo -e "\n${BLUE}📊 COMMIT SUMMARY${NC}"
echo "===================="
echo -e "${GREEN}✅ Successful commits: $commit_count${NC}"
if [[ $failed_commits -gt 0 ]]; then
    echo -e "${RED}❌ Failed commits: $failed_commits${NC}"
fi

# Push to remote
if [[ $commit_count -gt 0 ]]; then
    echo -e "\n${BLUE}🚀 Pushing changes to remote repository...${NC}"
    
    if git push origin main; then
        echo -e "${GREEN}✅ Successfully pushed all changes to GitHub!${NC}"
        
        # Show final status
        echo -e "\n${BLUE}📋 Final Repository Status:${NC}"
        git log --oneline -n $commit_count
        
    else
        echo -e "${RED}❌ Failed to push changes to remote repository${NC}"
        echo -e "${YELLOW}💡 You may need to pull first if there are remote changes${NC}"
        echo -e "${YELLOW}💡 Run: git pull origin main${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No commits were made, skipping push${NC}"
fi

# Cleanup
echo -e "\n${BLUE}🧹 Cleanup and Final Status${NC}"
echo "=============================="

# Show any remaining unstaged changes
if [[ -n "$(git status --porcelain)" ]]; then
    echo -e "${YELLOW}⚠️  Some files may still be unstaged (possibly large files):${NC}"
    git status --short
fi

echo -e "\n${GREEN}🎉 Automated commit and sync process completed!${NC}"

# Show excluded files summary
if [[ -f "$LARGE_FILES_LIST" ]] && [[ -s "$LARGE_FILES_LIST" ]]; then
    echo -e "\n${YELLOW}📋 Files excluded due to size (>90MB):${NC}"
    cat "$LARGE_FILES_LIST"
    echo -e "\n${YELLOW}💡 Consider using Git LFS for large files or exclude them permanently${NC}"
fi

exit 0