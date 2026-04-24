# 📂 Outputs Directory

This directory stores inference results, predictions, and evaluation artifacts.

## Expected Files

| File                | Description                                      |
| ------------------- | ------------------------------------------------ |
| `predictions.csv`   | Test set predictions (IMAGE → LABEL)             |

## Notes

- Output CSV files are excluded from version control via `.gitignore`.
- To generate predictions, run the full training + inference pipeline in `notebooks/`.
