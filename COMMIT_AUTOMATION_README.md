# 🤖 Automated Git Commit & Sync Scripts

This repository includes two automated scripts for managing git commits with intelligent file handling and meaningful commit messages.

## 📋 Scripts Overview

### 1. `auto_commit_sync.sh` - Main Automation Script
- **Purpose**: Commits each file individually with contextual commit messages
- **Features**: 
  - Excludes files larger than 90MB automatically
  - Generates meaningful commit messages based on file type and location
  - Pushes all commits to remote repository
  - Provides detailed progress and error reporting

### 2. `preview_commits.sh` - Dry Run Preview
- **Purpose**: Shows what files will be committed without making actual commits
- **Features**:
  - Lists all files to be committed with their sizes
  - Shows the exact commit messages that will be used
  - Identifies large files that will be excluded
  - Provides recommendations before running the main script

## 🚀 Usage

### Preview Mode (Recommended First)
```bash
./preview_commits.sh
```

### Full Commit and Sync
```bash
./auto_commit_sync.sh
```

## 📝 Commit Message Patterns

The script generates contextual commit messages following conventional commit standards:

| **File Type** | **Commit Pattern** | **Example** |
|---------------|-------------------|-------------|
| **Documentation** | `docs: action description` | `docs: add comprehensive getting started guide` |
| **Core Modules** | `feat: action module description` | `feat: add XGBoost model builder with GPU support` |
| **Configuration** | `config: action configuration` | `config: add system configuration with XGBoost settings` |
| **Dependencies** | `deps: action dependencies` | `deps: add Python dependencies for anomaly detection` |
| **Pipelines** | `pipeline: action pipeline description` | `pipeline: add training pipeline for model development` |
| **Analysis** | `analysis: action analysis description` | `analysis: add EDA analysis module for data insights` |
| **Tests** | `test: action test description` | `test: add ZenML/MLflow integration verification` |
| **Development** | `dev: action dev description` | `dev: add VS Code workspace configuration` |

## 🛡️ Safety Features

### Large File Protection
- **Automatic Detection**: Files >90MB are automatically excluded
- **GitHub Compatibility**: Prevents hitting GitHub's file size limits
- **Detailed Reporting**: Lists all excluded files with sizes
- **`.gitignore` Updates**: Automatically adds large files to `.gitignore`

### Error Handling
- **Individual File Commits**: If one file fails, others continue
- **Detailed Logging**: Shows success/failure for each commit
- **Recovery Information**: Provides guidance for failed operations
- **Rollback Capability**: Each file is a separate commit for easy rollback

## 📊 Expected Output

### Large Files Detected (90MB+)
```
❌ EXCLUDED: ./data/kddcup.data.corrected (709MB)
❌ EXCLUDED: ./anomaly/data/preprocessed_data_full.pkl (372MB)
```

### Successful Commits
```
✅ Successfully committed: src/data_ingester.py
   💬 feat: add KDD99 data ingestion module with strategy pattern

✅ Successfully committed: GETTING_STARTED.md  
   💬 docs: add comprehensive getting started guide with step-by-step instructions
```

### Final Summary
```
📊 COMMIT SUMMARY
==================
✅ Successful commits: 52
❌ Failed commits: 0
🚀 Successfully pushed all changes to GitHub!
```

## 🎯 Current Repository Status

Based on the latest analysis:
- **Files to commit**: 52 (1 modified + 51 new)
- **Large files excluded**: 3 files totaling ~1.8GB
- **Total upload size**: Manageable (<100MB after exclusions)
- **Commit strategy**: Individual commits for better tracking and rollback capability

## 💡 Best Practices

1. **Always run preview first**: `./preview_commits.sh`
2. **Review large file exclusions**: Consider using Git LFS for legitimate large files
3. **Monitor commit messages**: Ensure they accurately reflect your changes
4. **Check repository status**: Verify all important files are included

## 🔧 Customization

### Modify Size Limit
Edit `MAX_FILE_SIZE_MB=90` in both scripts to change the exclusion threshold.

### Custom Commit Messages
Modify the `generate_commit_message()` function to customize message patterns for your specific file types.

### Add File Type Recognition
Extend the case statements in `generate_commit_message()` to handle additional file types or patterns specific to your project.

---

**Note**: These scripts are designed specifically for the anomaly detection project structure but can be adapted for other projects by modifying the file pattern recognition logic.