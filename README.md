# CDFormer: Satellite Building Change Detection

**An end-to-end deep learning system for detecting urban building changes from bi-temporal satellite imagery.**  
Built for [Uzcosmos](https://uzcosmos.uz/) cadastral monitoring. Trained on Google Colab (T4 GPU), runs inference locally on Mac (Apple Silicon / MPS).

---

## Results

Best model evaluated on **LEVIR-CD** validation split (1,024 image pairs, 256×256 px):

| Metric    | Score  |
|-----------|--------|
| F1 Score  | 0.8835 |
| IoU       | 0.7913 |
| Precision | 0.8877 |
| Recall    | 0.8793 |

> Precision and recall are nearly equal — the model is well-balanced; not over-predicting and not being too conservative.

---

## Ablation Study

Each architectural component was removed one at a time to measure its contribution:

| Row | Model Variant                    | Best F1 |
|-----|----------------------------------|---------|
| 1   | Baseline SiamUNet                | 0.2602  |
| 2   | CDFormer + BGR only              | 0.7940  |
| 3   | Full CDFormer without BGR        | 0.8625  |
| 4   | Full CDFormer without TCDM       | 0.8268  |
| 5   | Full CDFormer without CSCP       | 0.8362  |
| 6   | Full CDFormer without LGC        | 0.8641  |
| 8   | Full CDFormer, no HNM            | 0.8625  |
| 9   | Full CDFormer + HNM              | ~0.836  |
| **Final** | **Full CDFormer, no HNM (15 epochs)** | **0.8836** |

---

## Architecture: CDFormer

CDFormer is a custom Siamese transformer architecture built on top of EfficientNet-B0. It uses **4-channel inputs** (RGB + Canny edge map) and includes five original components:

```
Before Image (T1) ──┐
                    ├──► Shared EfficientNet-B0 Backbone
After  Image (T2) ──┘
         │
    [4 scale features each]
         │
    ┌────▼─────────────────────────────────┐
    │  1. TCDM — Temporal Cross-Difference │  ← bidirectional cross-channel attention
    │  2. CSCP — Cross-Scale Propagation   │  ← top-down context between scales
    │  3. LGC  — Lightweight Global Ctx    │  ← transformer block at 1/32 scale
    │  4. BGR  — Boundary-Guided Refine    │  ← sharpens predictions at edges
    │  5. Building Aux Heads               │  ← per-timestep building presence
    └────────────────┬─────────────────────┘
                     │
              FPN Decoder
                     │
           Change Mask Output
```

### Component Details

| Component | What it does |
|-----------|-------------|
| **TCDM** (Temporal Cross-Difference Module) | Bidirectional cross-channel attention gated by a spatial difference signal. Replaces simple abs-diff with learned change representations. |
| **CSCP** (Cross-Scale Change Propagation) | Passes context from deeper (coarser) scales down to finer scales so high-level semantics guide local change detection. |
| **LGC** (Lightweight Global Context) | Single-layer multi-head self-attention at the deepest encoder scale (1/32). Provides long-range context at low cost. |
| **BGR** (Boundary-Guided Refinement) | Predicts building boundaries and feeds the boundary probability map back into the decoder to produce sharper edges. |
| **Building Aux Heads** | Per-timestep auxiliary classification heads that force the backbone to learn building-aware features, not just change features. |

### Loss Functions

- **FocalLovasz Loss** — primary change detection loss (handles class imbalance + directly optimises IoU)
- **Dice Loss** — boundary head (stable for sparse 1-pixel ring targets)
- **Contrastive Regularization** — cosine similarity pushes changed-region features apart
- **Edge-Weighted BCE** — extra supervision on building boundary pixels
- **Deep Supervision** — auxiliary heads at 1/8 and 1/4 scale

---

## Dataset: LEVIR-CD

| Split | Pairs |
|-------|-------|
| Train | 7,120 |
| Val   | 1,024 |

Images are 256×256 px patches cropped from 1024×1024 Google Earth images.  
Each pair has a binary change mask: `0 = unchanged`, `255 = changed building`.

**Download:** [LEVIR-CD on Google Drive](https://justchenhao.github.io/LEVIR/)  
Pre-patch the dataset to 256×256 before training (see `scripts/preprocess_dataset.py`).

---

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── cdformer.py          # Main model (CDFormer with ablation flags)
│   │   ├── siam_unet.py         # Baseline Siamese U-Net
│   │   └── transformer_cd.py    # Alternative TransformerCD model
│   ├── data/
│   │   ├── dataset.py           # ChangeDetectionDataset (4-channel input)
│   │   └── transforms.py        # Training & validation augmentations
│   ├── training/
│   │   ├── train.py             # Main training script
│   │   └── lightning_module.py  # PyTorch Lightning module + losses + metrics
│   ├── inference/
│   │   └── predictor.py         # Tiled inference with TTA and post-processing
│   ├── evaluation/
│   │   ├── cross_evaluate.py    # Evaluate F1/IoU/Precision/Recall on any split
│   │   └── generate_demo_visuals.py  # Generate comparison grid images
│   └── utils/
│       └── tiler.py             # Sliding-window tiler for large images
├── api/
│   └── main.py                  # FastAPI backend (REST endpoints)
├── web/
│   └── src/app/                 # Next.js + TailwindCSS frontend
├── scripts/
│   ├── preprocess_dataset.py    # Patch LEVIR-CD images to 256×256
│   └── run_inference.py         # CLI inference on a single image pair
├── configs/
│   └── default.yaml             # Hyperparameters and paths
├── checkpoints/                 # Saved .ckpt weight files
├── outputs/                     # Demo visuals and logs
├── notebooks/
│   └── Colab_Instructions.md    # Step-by-step Colab training guide
├── change_detection.ipynb       # Original experiment notebook
├── continuation_change_detection.ipynb  # Ablation study notebook
├── requirements.txt
└── run_demo.sh                  # One-command launcher (backend + frontend)
```

---

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/dilrabo815/uzcosmos-change-detection.git
cd uzcosmos-change-detection
pip install -r requirements.txt
```

### 2. Download Best Checkpoint

Place your trained `.ckpt` file inside the `checkpoints/` folder.  
The filename should contain `cdformer` for the model to be auto-detected.

```
checkpoints/
└── cdformer_TCDM_CSCP_BGR_LGC_BH-epoch=10-val_f1=0.8836.ckpt
```

### 3. Run the Web Demo

```bash
./run_demo.sh
```

Open `http://localhost:3000` — upload a before/after satellite image pair and see the change overlay in real time.

To stop: press `CTRL+C`.

---

## Training from Scratch (Google Colab)

### Step 1 — Prepare Files

1. Zip your `src/` folder → `src.zip`
2. Pre-patch LEVIR-CD to 256×256 → `levir_256.zip`
3. Upload both to Google Drive under `MyDrive/Uzcosmos_Project/`

### Step 2 — Open Colab

Use the provided notebook: `continuation_change_detection.ipynb`  
Set runtime to **T4 GPU**.

### Step 3 — Run Training

```bash
# Mount Drive and extract code
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/Uzcosmos_Project/src.zip /content/
!unzip -qo /content/src.zip -d /content/
!unzip -q /content/drive/MyDrive/Uzcosmos_Project/levir_256.zip -d /content/data/ready/

# Install dependencies
!pip install pytorch-lightning albumentations timm opencv-python-headless torchmetrics

# Train full CDFormer (15 epochs)
%env PYTHONPATH=/content
!python src/training/train.py \
  --model custom \
  --data_dirs /content/data/ready/levir_256 \
  --epochs 15 \
  --batch_size 8 \
  --patience 5 \
  --checkpoint_dir /content/drive/MyDrive/Uzcosmos_Project/checkpoints/final_run/
```

### Training CLI Flags

| Flag | Description |
|------|-------------|
| `--model custom` | Use CDFormer (full model) |
| `--model baseline` | Use SiamUNet (lightweight baseline) |
| `--epochs N` | Max training epochs |
| `--batch_size N` | Batch size (8 recommended for Colab T4) |
| `--patience N` | Early stopping patience |
| `--hard_negative_mining` | Oversample near-building patches 2.5× |
| `--no_tcdm` | Ablation: replace TCDM with simple abs-diff |
| `--no_cscp` | Ablation: disable cross-scale context |
| `--no_bgr` | Ablation: disable boundary refinement |
| `--no_lgc` | Ablation: disable global context |
| `--no_build_heads` | Ablation: disable building aux heads |

---

## Running the Ablation Study

Each ablation row runs as a separate training job. The `continuation_change_detection.ipynb` notebook has all rows pre-configured. To run a specific row:

```bash
# Row 1 — Baseline (SiamUNet)
python src/training/train.py --model baseline --epochs 5 --batch_size 8

# Row 2 — CDFormer + BGR only
python src/training/train.py --model custom --no_tcdm --no_cscp --no_lgc --no_build_heads --epochs 5

# Row 3 — Full CDFormer without BGR
python src/training/train.py --model custom --no_bgr --epochs 5

# Row 4 — Full CDFormer without TCDM
python src/training/train.py --model custom --no_tcdm --epochs 5

# Full model (no HNM)
python src/training/train.py --model custom --epochs 15
```

---

## Evaluation

Evaluate any checkpoint on any dataset split:

```bash
python src/evaluation/cross_evaluate.py \
  --model_type custom \
  --checkpoint checkpoints/cdformer_TCDM_CSCP_BGR_LGC_BH-epoch=10-val_f1=0.8836.ckpt \
  --data_dirs data/ready/levir_256 \
  --batch_size 8
```

Output:

```
| Dataset Path           | F1 Score | IoU    | Precision | Recall |
|------------------------|----------|--------|-----------|--------|
| data/ready/levir_256   | 0.8835   | 0.7913 | 0.8877    | 0.8793 |
```

---

## Inference on a Custom Image Pair

```bash
python scripts/run_inference.py \
  --checkpoint checkpoints/cdformer_TCDM_CSCP_BGR_LGC_BH-epoch=10-val_f1=0.8836.ckpt \
  --image_A path/to/before.png \
  --image_B path/to/after.png \
  --output outputs/result/
```

The predictor uses:
- **Tiled inference** — 256×256 tiles with 64px overlap (handles any image size)
- **TTA** — test-time augmentation (horizontal/vertical flips averaged)
- **Change gate** — suppresses predictions in areas with no structural difference
- **Morphological post-processing** — removes small noise regions

---

## API Endpoints

The FastAPI backend (`api/main.py`) exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check that the model loaded correctly |
| `/predict` | POST | Upload `file_A` + `file_B`, get change mask + overlay |
| `/metrics` | GET | Model configuration info |
| `/samples` | GET | List available demo visuals |

Start backend only:
```bash
source venv/bin/activate
uvicorn api.main:app --port 8000
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 16
  img_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 15

augmentation:
  horizontal_flip: true
  vertical_flip: true
  rotate_90: true
  color_jitter: { brightness: 0.3, contrast: 0.3 }
  sensor_degradation: { gauss_noise_var: [10, 50], blur_limit: [3, 7] }

inference:
  tile_size: 256
  overlap: 64
  threshold: 0.5
  device: auto   # auto-detects CUDA / MPS / CPU
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Model backbone | EfficientNet-B0 via `timm` |
| Training framework | PyTorch Lightning |
| Augmentations | Albumentations |
| Metrics | TorchMetrics |
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js + TailwindCSS |
| Inference device | CUDA (Colab) / MPS (Mac M2) / CPU |

---

## Requirements

```
fastapi, uvicorn, python-multipart
torch, torchvision
pytorch-lightning
torchmetrics
albumentations
opencv-python-headless
numpy, matplotlib, scikit-learn
einops, timm
pydantic, gdown
```

Install: `pip install -r requirements.txt`

---

## License

MIT License — free to use, modify, and distribute.
