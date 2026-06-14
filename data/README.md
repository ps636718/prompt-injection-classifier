# 📊 Data

This directory holds the training and test CSV files used by the classifier.

## Required Files

| File | Description |
|------|-------------|
| `merged_train_70.csv` | Training set — 70% of the full dataset |
| `merged_test_30.csv`  | Test set — 30% of the full dataset |

## Schema

| Column | Type | Required | Description |
|--------|------|:--------:|-------------|
| `Prompt` | string | ✅ | Primary text prompt |
| `isMalicious` | int 0/1 | ✅ train | Ground-truth label |
| `question1` | string | ➖ | Secondary text column (merged if present) |
| `question2` | string | ➖ | Tertiary text column (merged if present) |
| `Length` | float | ➖ | Pre-computed prompt length |
| `Perplexity` | float | ➖ | Language model perplexity |

## Note

Data files are **not included** in this repository (excluded via `.gitignore`).  
Suitable open datasets:
-Dataset - https://www.kaggle.com/competitions/data-sprint-prompt-based-classififcation/data
