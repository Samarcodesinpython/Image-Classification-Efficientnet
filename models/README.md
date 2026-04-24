# 📂 Models Directory

This directory stores trained model checkpoints.

## Expected Files

| File         | Description                                      |
| ------------ | ------------------------------------------------ |
| `model.pth`  | Best EfficientNet-B2 checkpoint (saved on best validation accuracy) |

## Notes

- Model checkpoints (`.pth`, `.pt`) are excluded from version control via `.gitignore` due to file size.
- To reproduce the checkpoint, run the training notebook in `notebooks/`.
- The saved checkpoint contains only `state_dict` (model weights), not the full model object.
