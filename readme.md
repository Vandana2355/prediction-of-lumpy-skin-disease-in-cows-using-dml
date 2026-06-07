# LSD Classification Model (ViT-Based)

A Vision Transformer (ViT) based image classifier optimized for GTX1650 (4GB VRAM) and 8GB RAM systems. The model classifies images and provides severity scores (0-10).

## Features

- **Binary classification** with severity scoring
- **Memory optimized** for low-end GPUs
- **Gradient accumulation** to simulate larger batch sizes
- **Mixed precision training** (AMP) for faster training
- **Auto-labeling** from filename patterns

## Dataset Structure

All images should be in a single folder:

```
data/
├── Normal_001.jpg     # Label: 0 (Normal)
├── Normal_002.png     # Label: 0 (Normal)
├── Abnormal_001.jpg   # Label: 1 (Abnormal)
├── LSD_001.png        # Label: 1 (Abnormal)
└── ...
```

**Labeling Rule**: Files starting with "Normal" (case-insensitive) → label 0, all others → label 1

## Installation

1. **Clone or download** this repository

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify PyTorch installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True` if GPU is available.

## Usage

### Training

1. **Edit `main.py`**: Set `DATA_ROOT` to your image folder path
   ```python
   DATA_ROOT = "data/"  # Change this to your folder
   ```

2. **Run training**:
   ```bash
   python main.py
   ```

3. **Output**: 
   - `best_vit_lsd_classifier.pth` (saved model)
   - Training logs in console

### Evaluation

Check model performance on validation set:

```bash
python evaluate_model.py
```

**Output includes**:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (visual)
- ROC Curve with AUC score
- Per-image predictions with severity scores
- Sample predictions saved to `evaluation_results.txt`

### Inference on New Images

```python
from main import infer_image, create_vit_model
import torch

# Load model
model = create_vit_model().to('cuda')
model.load_state_dict(torch.load('best_vit_lsd_classifier.pth'))

# Predict
prob, severity, category = infer_image(model, 'path/to/image.jpg')
print(f"Probability: {prob:.3f}")
print(f"Severity: {severity:.2f}/10")
print(f"Category: {category}")  # Mild / Moderate / Severe
```

## Configuration

Key parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_ROOT` | `"data/"` | Path to image folder |
| `IMG_SIZE` | `224` | Input image size |
| `EPOCHS` | `12` | Training epochs |
| `BATCH_SIZE` | `2` | Micro-batch size (VRAM safe) |
| `ACCUM_STEPS` | `4` | Gradient accumulation steps |
| `BASE_LR` | `2e-5` | Learning rate |
| `NUM_WORKERS` | `2` | DataLoader workers |

**Effective batch size** = `BATCH_SIZE × ACCUM_STEPS` = 8

## Severity Scale

| Probability | Severity | Category |
|-------------|----------|----------|
| 0.0 - 0.4 | 0 - 4 | **Mild** |
| 0.4 - 0.7 | 4 - 7 | **Moderate** |
| 0.7 - 1.0 | 7 - 10 | **Severe** |

## Model Architecture

- **Base Model**: ViT-Base/16 (pretrained on ImageNet-21k)
- **Fine-tuning**: Last 4 transformer blocks + classification head
- **Output**: Single logit (binary classification with BCEWithLogitsLoss)
- **Total Params**: ~86M (only ~25M trainable)

## Memory Optimization Features

1. **Small batch size** (2) with gradient accumulation
2. **Mixed precision training** (FP16)
3. **Limited workers** for DataLoader
4. **Memory clearing** after each epoch
5. **Frozen backbone** (only fine-tune top layers)

## Troubleshooting

**Out of Memory Error**:
- Reduce `BATCH_SIZE` to 1
- Reduce `NUM_WORKERS` to 0 or 1
- Reduce `IMG_SIZE` to 192

**Slow Training**:
- Increase `NUM_WORKERS` if you have spare RAM
- Ensure CUDA is available: check `DEVICE` output

**Poor Accuracy**:
- Ensure dataset is balanced (similar number of normal/abnormal)
- Increase `EPOCHS` (try 20-30)
- Check if filenames follow the naming convention
- Verify image quality

## Requirements


- **RAM**: 8GB minimum
- **Python**: 3.8+


## Files

- `main.py` - Training script
- `evaluate_model.py` - Model evaluation script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## License

This project is provided as-is for educational and research purposes.
