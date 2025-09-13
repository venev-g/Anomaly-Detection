#!/bin/bash

# Preview Script for Automated Git Commits
# Shows what files will be committed and with what messages (dry run)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MAX_FILE_SIZE_MB=90
REPO_DIR="/workspaces/Anomaly-Detection"

echo -e "${CYAN}👀 PREVIEW: Automated Git Commit Analysis${NC}"
echo "=============================================="

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

# Function to get human readable file size with units
get_file_size_human() {
    local file="$1"
    if [[ -f "$file" ]]; then
        ls -lh "$file" | awk '{print $5}'
    else
        echo "0B"
    fi
}

# Same commit message generation function as main script
generate_commit_message() {
    local file="$1"
    local action="$2"
    local filename=$(basename "$file")
    local dirname=$(dirname "$file")
    local extension="${filename##*.}"
    
    case "$file" in
        "README.md"|"*/README.md")
            echo "docs: ${action} project README with setup and usage instructions"
            ;;
        "GETTING_STARTED.md"|"*/GETTING_STARTED.md")
            echo "docs: ${action} comprehensive getting started guide with step-by-step instructions"
            ;;
        "QUICKSTART.md"|"*/QUICKSTART.md")
            echo "docs: ${action} quick start guide for rapid deployment"
            ;;
        "config.yaml"|"*/config.yaml")
            echo "config: ${action} system configuration with XGBoost and ZenML settings"
            ;;
        "requirements.txt"|"*/requirements.txt")
            echo "deps: ${action} Python dependencies for anomaly detection system"
            ;;
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
        "steps/"*".py")
            local step_name=$(basename "$file" .py)
            echo "pipeline: ${action} ZenML pipeline step - ${step_name}"
            ;;
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
        "analysis/"*".py")
            echo "analysis: ${action} EDA analysis module for data insights"
            ;;
        "analysis/"*".ipynb")
            echo "analysis: ${action} comprehensive EDA Jupyter notebook for KDD99 dataset"
            ;;
        "explanations/"*".py")
            local pattern_name=$(basename "$file" .py)
            echo "docs: ${action} design pattern example - ${pattern_name}"
            ;;
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
        ".vscode/"*)
            echo "dev: ${action} VS Code workspace configuration and settings"
            ;;
        "data/"*|"*/data/"*)
            echo "data: ${action} dataset file for anomaly detection (WARNING: Check file size)"
            ;;
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

# Function to check if file should be excluded
should_exclude_file() {
    local file="$1"
    local size_mb=$(get_file_size_mb "$file")
    
    if [[ $size_mb -gt $MAX_FILE_SIZE_MB ]]; then
        return 0  # Should exclude
    else
        return 1  # Should include
    fi
}

# Analyze large files
echo -e "${YELLOW}🔍 Large Files Analysis (>90MB):${NC}"
large_files_found=false
while IFS= read -r -d '' file; do
    if should_exclude_file "$file"; then
        size_human=$(get_file_size_human "$file")
        echo -e "${RED}  ❌ EXCLUDED: $file ($size_human)${NC}"
        large_files_found=true
    fi
done < <(find . -type f -print0)

if [[ "$large_files_found" == "false" ]]; then
    echo -e "${GREEN}  ✅ No large files found${NC}"
fi

# Analyze files to be committed
echo -e "\n${BLUE}📋 Files to be Committed:${NC}"

# Get modified files
modified_count=0
untracked_count=0
total_size=0

echo -e "\n${CYAN}📝 Modified Files:${NC}"
if git diff --name-only | grep -v "^$" > /dev/null 2>&1; then
    while IFS= read -r file; do
        if [[ -f "$file" ]] && ! should_exclude_file "$file"; then
            size_human=$(get_file_size_human "$file")
            size_mb=$(get_file_size_mb "$file")
            total_size=$((total_size + size_mb))
            commit_msg=$(generate_commit_message "$file" "update")
            echo -e "${GREEN}  ✅ $file ($size_human)${NC}"
            echo -e "${BLUE}     💬 $commit_msg${NC}"
            ((modified_count++))
        fi
    done < <(git diff --name-only)
else
    echo -e "${YELLOW}  📭 No modified files${NC}"
fi

echo -e "\n${CYAN}📁 New/Untracked Files:${NC}"
if git ls-files --others --exclude-standard | grep -v "^$" > /dev/null 2>&1; then
    while IFS= read -r file; do
        if [[ -f "$file" ]] && ! should_exclude_file "$file"; then
            size_human=$(get_file_size_human "$file")
            size_mb=$(get_file_size_mb "$file")
            total_size=$((total_size + size_mb))
            commit_msg=$(generate_commit_message "$file" "add")
            echo -e "${GREEN}  ✅ $file ($size_human)${NC}"
            echo -e "${BLUE}     💬 $commit_msg${NC}"
            ((untracked_count++))
        fi
    done < <(git ls-files --others --exclude-standard)
else
    echo -e "${YELLOW}  📭 No untracked files${NC}"
fi

# Summary
echo -e "\n${CYAN}📊 COMMIT PREVIEW SUMMARY${NC}"
echo "=========================="
echo -e "${GREEN}📝 Modified files to commit: $modified_count${NC}"
echo -e "${GREEN}📁 New files to commit: $untracked_count${NC}"
echo -e "${GREEN}📋 Total files to commit: $((modified_count + untracked_count))${NC}"
echo -e "${GREEN}💾 Total size to upload: ${total_size}MB${NC}"

# Recommendations
echo -e "\n${CYAN}💡 Recommendations:${NC}"
if [[ $((modified_count + untracked_count)) -eq 0 ]]; then
    echo -e "${YELLOW}  ⚠️  No files to commit - repository may be up to date${NC}"
elif [[ $total_size -gt 500 ]]; then
    echo -e "${YELLOW}  ⚠️  Large upload size (${total_size}MB) - consider reviewing files${NC}"
elif [[ $((modified_count + untracked_count)) -gt 50 ]]; then
    echo -e "${YELLOW}  ⚠️  Many files to commit (${modified_count + untracked_count}) - this will create many commits${NC}"
else
    echo -e "${GREEN}  ✅ Ready to commit - reasonable number of files and size${NC}"
fi

# Show git status for reference
echo -e "\n${CYAN}📋 Current Git Status:${NC}"
git status --short

echo -e "\n${CYAN}🚀 To proceed with actual commits, run:${NC}"
echo -e "${GREEN}  ./auto_commit_sync.sh${NC}"

exit 0