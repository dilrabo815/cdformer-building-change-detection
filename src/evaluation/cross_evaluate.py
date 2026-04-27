import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall

from src.models.siam_unet import SiamUNet
from src.models.transformer_cd import TransformerCD
from src.models.cdformer import CDFormer
from src.training.lightning_module import CDLightningModule
from src.data.dataset import ChangeDetectionDataset
from src.data.transforms import get_validation_transforms

def evaluate_dataset(model, dataloader, device):
    model.eval()

    # Epoch-level metrics — accumulate TP/FP/FN across all batches, compute once at the end
    f1_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)

            logits = model(img_A, img_B)
            probs = torch.sigmoid(logits)

            targets_int = label.long()
            f1_metric.update(probs.view(-1), targets_int.view(-1))
            iou_metric.update(probs.view(-1), targets_int.view(-1))
            precision_metric.update(probs.view(-1), targets_int.view(-1))
            recall_metric.update(probs.view(-1), targets_int.view(-1))

    return {
        "F1": f1_metric.compute().item(),
        "IoU": iou_metric.compute().item(),
        "Precision": precision_metric.compute().item(),
        "Recall": recall_metric.compute().item(),
    }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # in_channels=4: RGB + Canny edge boundary map (matches updated dataset/training)
    if args.model_type == 'baseline':
        base_model = SiamUNet(in_channels=4, classes=1)
    elif args.model_type == 'advanced':
        base_model = TransformerCD(in_channels=4, classes=1)
    else:
        base_model = CDFormer(in_channels=4, classes=1)

    model = CDLightningModule.load_from_checkpoint(args.checkpoint, model=base_model)
    model.to(device)

    print(f"Loaded checkpoint from: {args.checkpoint} on {device}")

    results = {}
    for dataset_dir in args.data_dirs:
        print(f"\n--- Evaluating on dataset: {dataset_dir} ---")
        dataset = ChangeDetectionDataset(dataset_dir, subset="test", transform=get_validation_transforms(args.img_size))

        if len(dataset) == 0:
            print(f"Warning: 'test' split not found, using 'val' split for {dataset_dir}")
            dataset = ChangeDetectionDataset(dataset_dir, subset="val", transform=get_validation_transforms(args.img_size))

        if len(dataset) == 0:
            print(f"Skipping {dataset_dir}: No images found.")
            continue

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        metrics = evaluate_dataset(model, dataloader, device)
        results[dataset_dir] = metrics

    # Print Markdown Table
    print("\n## Cross-Dataset Evaluation Results\n")
    print("| Dataset Path | F1 Score | IoU | Precision | Recall |")
    print("|---|---|---|---|---|")
    for name, m in results.items():
        print(f"| {name} | {m['F1']:.4f} | {m['IoU']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CD model across multiple datasets to test generalization.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .ckpt file")
    parser.add_argument("--model_type", type=str, choices=["baseline", "advanced", "custom"], default="custom")
    parser.add_argument("--data_dirs", type=str, nargs="+", default=["data/ready/levir"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)

    args = parser.parse_args()
    main(args)
