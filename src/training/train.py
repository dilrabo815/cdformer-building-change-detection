import os
import sys

# Ensure project root is on Python path (handles Colab, Mac, any working directory)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from src.data.dataset import ChangeDetectionDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.siam_unet import SiamUNet
from src.models.transformer_cd import TransformerCD
from src.models.cdformer import CDFormer
from src.training.lightning_module import CDLightningModule


def _get_experiment_name(args) -> str:
    """Auto-generate a short descriptive name from the CLI flags for this run."""
    if args.model == "baseline":
        return "M0_baseline_siamunet"
    if args.model == "advanced":
        return "M0_advanced_transformercd"
    parts = ["cdformer"]
    if not args.no_tcdm:        parts.append("TCDM")
    if not args.no_cscp:        parts.append("CSCP")
    if not args.no_bgr:         parts.append("BGR")
    if not args.no_lgc:         parts.append("LGC")
    if not args.no_build_heads: parts.append("BH")
    if args.hard_negative_mining: parts.append("HNM")
    return "_".join(parts)


def _save_config(args, experiment_name: str):
    """Write a JSON config file alongside the checkpoints for reproducibility."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config = vars(args).copy()
    config['experiment_name'] = experiment_name
    config_path = os.path.join(args.checkpoint_dir, f"{experiment_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved experiment config → {config_path}")
    return config_path


def _build_weighted_sampler(datasets: list) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler from per-dataset sample weights.

    Hard-negative mining strategy (from ChangeDetectionDataset.get_sample_weights):
        - Patches with little-but-nonzero change (near buildings, hard negatives)
          are oversampled at 2.5× to teach the model to ignore appearance changes
          on stable buildings.
        - Pure background patches are undersampled (0.6×).
        - Clearly changed patches are normal weight (1.0×).
    """
    all_weights = []
    for ds in datasets:
        # ConcatDataset stores individual datasets in .datasets
        if hasattr(ds, 'get_sample_weights'):
            all_weights.extend(ds.get_sample_weights())
        else:
            all_weights.extend([1.0] * len(ds))

    weights_tensor = torch.tensor(all_weights, dtype=torch.float32)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


def main(args):
    pl.seed_everything(42)

    experiment_name = _get_experiment_name(args)
    print(f"[Train] Experiment: {experiment_name}")
    _save_config(args, experiment_name)

    train_datasets = []
    val_datasets   = []

    for d_dir in args.data_dirs:
        train_datasets.append(ChangeDetectionDataset(
            root_dir=d_dir,
            subset="train",
            transform=get_training_transforms(args.img_size),
        ))
        val_datasets.append(ChangeDetectionDataset(
            root_dir=d_dir,
            subset="val",
            transform=get_validation_transforms(args.img_size),
        ))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset   = ConcatDataset(val_datasets)

    # Hard-negative mining: weighted sampler oversamples near-building negatives
    if args.hard_negative_mining:
        print("[Train] Hard-negative mining enabled — building WeightedRandomSampler")
        sampler = _build_weighted_sampler(train_datasets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,                # shuffle=False when using sampler
            num_workers=args.num_workers,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model selection
    if args.model == "baseline":
        model = SiamUNet(in_channels=4, classes=1)
    elif args.model == "advanced":
        model = TransformerCD(in_channels=4, classes=1)
    else:
        # CDFormer with per-component ablation flags.
        # Pass --no_tcdm / --no_cscp / --no_bgr / --no_lgc / --no_build_heads
        # to disable individual components for the ablation study.
        model = CDFormer(
            in_channels=4, classes=1,
            use_tcdm=not args.no_tcdm,
            use_cscp=not args.no_cscp,
            use_bgr=not args.no_bgr,
            use_lgc=not args.no_lgc,
            use_build_heads=not args.no_build_heads,
        )
        enabled = [c for c, v in [("TCDM", not args.no_tcdm), ("CSCP", not args.no_cscp),
                                    ("BGR", not args.no_bgr), ("LGC", not args.no_lgc),
                                    ("BuildHeads", not args.no_build_heads)] if v]
        print(f"[CDFormer] Active components: {', '.join(enabled)}")

    lightning_cd = CDLightningModule(
        model,
        learning_rate=args.lr,
        experiment_name=experiment_name,
        ablation_config=vars(args),
    )

    # Callbacks
    # For CDFormer (custom) we monitor val_best_f1_calibrated — the F1 at the
    # sweep-optimal threshold rather than the fixed 0.5 default.  For other
    # models we fall back to val_f1.
    #
    # Note: val_best_f1_calibrated is logged in on_validation_epoch_end and is
    # guaranteed to exist after every validation epoch (including the sanity
    # check at epoch 0), so this is safe to monitor from the first epoch.
    monitor_metric = "val_best_f1_calibrated" if args.model == "custom" else "val_f1"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=experiment_name + "-{epoch:02d}-val_f1={val_f1:.4f}",
        save_top_k=3,
        monitor=monitor_metric,
        mode="max",
    )
    early_stop = EarlyStopping(monitor=monitor_metric, patience=args.patience,
                                mode="max", strict=False)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stop],
        default_root_dir=args.checkpoint_dir,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    ckpt_path = args.resume_from if args.resume_from else None
    trainer.fit(lightning_cd,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)

    print("Training Finished. Best checkpoint:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Building Change Detection.")
    parser.add_argument("--data_dirs",  type=str, nargs="+",
                        default=["data/ready/levir"],
                        help="Paths to standardized datasets")
    parser.add_argument("--model", type=str,
                        choices=["baseline", "advanced", "custom"], default="custom")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int, default=15)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--resume_from",    type=str, default=None)
    parser.add_argument("--hard_negative_mining", action="store_true",
                        help="Oversample hard negatives (near-building unchanged patches)")
    # ── Ablation flags (CDFormer only) ───────────────────────────────────────
    parser.add_argument("--no_tcdm",        action="store_true",
                        help="Replace TCDM with simple abs-diff (ablation)")
    parser.add_argument("--no_cscp",        action="store_true",
                        help="Disable cross-scale context propagation (ablation)")
    parser.add_argument("--no_bgr",         action="store_true",
                        help="Disable Boundary-Guided Refinement (ablation)")
    parser.add_argument("--no_lgc",         action="store_true",
                        help="Disable Lightweight Global Context (ablation)")
    parser.add_argument("--no_build_heads", action="store_true",
                        help="Disable per-timestep building aux heads (ablation)")
    args = parser.parse_args()
    main(args)
