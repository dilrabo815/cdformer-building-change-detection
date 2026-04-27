# Training on Google Colab (LEVIR-CD Workflow)

Follow these steps to train your building change detection model using a free T4 GPU.

### Step 1: Prepare Files
1. Zip your local `src` folder as `src.zip`.
2. Upload `src.zip`, `train.zip`, `val.zip`, and `test.zip` to your Google Drive in a folder named `Uzcosmos_Project`.

### Step 2: Colab Setup
1. Create a new notebook on [Colab](https://colab.research.google.com/).
2. Set Runtime to **T4 GPU**.

### Step 3: Run Execution Cells

**Cell 1: Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2: Extract Code and Data**
```bash
!cp /content/drive/MyDrive/Uzcosmos_Project/src.zip /content/
!cp /content/drive/MyDrive/Uzcosmos_Project/train.zip /content/
!cp /content/drive/MyDrive/Uzcosmos_Project/val.zip /content/
!cp /content/drive/MyDrive/Uzcosmos_Project/test.zip /content/

!unzip -qo src.zip
!mkdir -p data/ready/levir/train data/ready/levir/val data/ready/levir/test

!unzip -qo train.zip -d data/ready/levir/train
!unzip -qo val.zip -d data/ready/levir/val
!unzip -qo test.zip -d data/ready/levir/test

# Mandatory package initialization
!touch src/__init__.py src/data/__init__.py src/models/__init__.py src/training/__init__.py src/inference/__init__.py src/evaluation/__init__.py src/utils/__init__.py
```

**Cell 3: Install & Start Training**
```bash
!pip install pytorch-lightning albumentations timm opencv-python-headless
%env PYTHONPATH=/content

!python src/training/train.py \
  --data_dirs /content/data/ready/levir \
  --model baseline \
  --epochs 100 \
  --batch_size 16 \
  --checkpoint_dir /content/drive/MyDrive/Uzcosmos_Project/checkpoints/
```
