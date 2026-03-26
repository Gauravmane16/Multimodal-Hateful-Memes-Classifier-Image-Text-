# Multimodal Hateful Meme Classifier

A deep learning model that combines vision and language understanding to detect hateful memes. This project implements a multimodal fusion architecture using ResNet50 for images and MiniLM for text, with integrated OCR capabilities.

**Final Fusion Model achieved 56.52% Accuracy and 42.50% ROC-AUC on the test set.**

## 🎯 Overview

**Final Model**: Multimodal Fusion (ResNet50 + MiniLM)
- **Accuracy**: 56.52%
- **ROC-AUC**: 42.50%

## 🏗️ Architecture

### Components

## ⚖️ Decisions

- Image encoder: `ResNet50` (pretrained ImageNet weights) for robust visual features.
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2` for compact, fast text embeddings.
- OCR: `EasyOCR` to extract text from images (downloaded models at first run).
- Fusion: Concatenate image + text embeddings → MLP (hidden 512) for simplicity and interpretability.
- Loss & imbalance: `BCEWithLogitsLoss` with `pos_weight=1.5` to penalize false negatives.
- Training: AdamW, AMP enabled, early stopping (patience=3).

### Baseline Models
1. **Image-only (ResNet50)**: CNN features only
2. **Text-only (MiniLM)**: Caption + OCR text only
3. **Multimodal Fusion**: Combined embeddings

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- torch, torchvision
- transformers, sentence-transformers
- fastapi, uvicorn
- scikit-learn
- easyocr
- pillow, pyyaml

## 📂 Project Structure

```
hateful-memes-mm/
├── data/
│   ├── raw/                  # Original images
│   ├── processed/            # Preprocessed images (optional)
│   └── splits.json           # Train/val/test splits
├── src/
│   ├── data.py               # Dataset loading & preprocessing
│   ├── models.py             # Model architectures
│   ├── train.py              # Training loop
│   ├── eval.py               # Evaluation metrics
│   ├── infer.py              # Inference utilities
│   └── utils.py              # Helpers (seed, etc.)
├── app.py                    # FastAPI server
├── config.yaml               # Training hyperparameters
├── requirements.txt
├── README.md
├── report.md                 # Results & analysis
├── metrics.json              # Test set metrics (generated)
└── checkpoints/
    └── fusion.pt             # Best model checkpoint (generated)
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
seed: 42
data:
  image_size: 224
  batch_size: 32
model:
  image_encoder: "resnet50"
  text_encoder: "all-MiniLM-L6-v2"
  fusion_hidden_dim: 512
  dropout: 0.3
training:
  lr: 3e-5
  epochs: 10
  weight_decay: 0.01
  early_stopping_patience: 3
loss:
  pos_weight: 1.5
paths:
  model_ckpt: "checkpoints/fusion.pt"
```

## 📊 Dataset Format

`data/splits.json` structure:

```json
{
  "train": [
    {
      "id": "img_1",
      "image": "data/raw/img_1.jpg",
      "caption": "Meme caption text",
      "label": 0
    }
  ],
  "val": [...],
  "test": [...]
}
```

- `label`: 0 (not_hateful) or 1 (hateful)
- Images should be in `data/raw/`

## 🚀 Usage

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the Makefile
make install
```

### Training

```bash
# Run training pipeline
python -m src.train

# Or use Makefile
make train
```

This will:
1. Load train/val/test splits from `data/splits.json`
2. Initialize encoders and fusion model
3. Train with early stopping (patience=3)
4. Save best model to `checkpoints/fusion.pt`
5. Evaluate on test set
6. Output metrics to `metrics.json`

### Expected Output

After training completes (example — final run for this project):

```
============================================================
FINAL TEST RESULTS
============================================================
Accuracy:  56.52%
Precision: 25.00%
Recall:    12.50%
F1-Score:  16.67%
ROC-AUC:   42.50%

Confusion Matrix (%):
[[80.0, 20.0]
 [87.5, 12.5]]
============================================================
Metrics saved to metrics.json
```

### API Inference

```bash
# Start server
uvicorn app:app --reload

# In another terminal, test the endpoint
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@test.jpg" \
  -F "caption=Sample meme caption"

# Response:
# {
#   "label": "hateful",
#   "confidence": 0.7834
# }
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## 📈 Evaluation Metrics

All metrics are reported on the **test set** as percentages:

| Model | Accuracy % | Precision % | Recall % | F1 % | ROC-AUC % |
|-------|-----------|-------------|----------|------|-----------|
| Image-only (ResNet50) | TBD | TBD | TBD | TBD | TBD |
| Text-only (MiniLM) | TBD | TBD | TBD | TBD | TBD |
| Fusion (concat) | TBD | TBD | TBD | TBD | TBD |

### Confusion Matrix (% normalized by true class)

```
[[TN%  FP%]
 [FN%  TP%]]
```

Example:
```
[[74.1  25.9]
 [18.6  81.4]]
```

## 🔍 Key Implementation Details

### Training
- **Optimizer**: AdamW (lr=3e-5, weight_decay=0.01)
- **Loss**: BCEWithLogitsLoss with pos_weight=1.5 for class imbalance
- **AMP**: Automatic Mixed Precision (torch.amp) for faster training
- **Early Stopping**: Monitor validation accuracy, patience=3
- **Data Augmentation**: Random horizontal flip, rotation, normalization

### Inference
1. Resize image to 224×224
2. Extract OCR text from image
3. Combine caption + OCR text
4. Tokenize to 128 tokens
5. Forward through image & text encoders
6. Concatenate embeddings
7. Fusion network → logits
8. Sigmoid(logits) → confidence score

### Class Imbalance Handling
- `pos_weight=1.5` in loss function
- Proper metric computation with normalized confusion matrix

## 📝 Ablation Studies

(To be added after baseline training)

1. **Without OCR**: Compare caption-only vs. caption+OCR
2. **Without Data Augmentation**: Compare augmented vs. raw images
3. **Fusion Strategy**: Concatenation vs. other fusion methods
4. **Model Size**: ResNet50 vs. smaller encoders

## 🐛 Troubleshooting

### GPU not available
- Model will automatically fall back to CPU
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Empty splits.json
- Populate `data/splits.json` with your dataset following the format above
- Ensure image paths are relative and files exist

### Module import errors
- Run scripts from project root: `python -m src.train`
- Not as: `python src/train.py`

### OCR is slow
- First run downloads model, subsequent runs are cached
- Can be disabled in `data.py` if needed

## 📚 References

- CLIP: https://openai.com/research/learning-transferable-models-for-computational-linguistics
- Sentence-Transformers: https://www.sbert.net/
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Hateful Memes Detection Project

---

## Makefile Commands

```bash
make install          # Install dependencies
make train            # Run training
make eval             # Run evaluation
make infer            # Run inference
make clean            # Remove cache files
```

## ✅ Checklist

- [x] Modular code architecture
- [x] Configuration management (YAML)
- [x] Multimodal fusion model
- [x] Train/val/test pipeline
- [x] Proper metrics computation
- [x] Early stopping & checkpointing
- [x] FastAPI inference server
- [x] Complete documentation
- [ ] Test set evaluation (run training first)
- [ ] Misclassified examples analysis (to be added)
- [ ] Report with results (to be filled)
