"""
Local Inference Script for MacBook Pro M2.

Run inference on any before/after image pair using a trained checkpoint,
without needing the API server.

Usage:
    python scripts/run_inference.py \
        --checkpoint checkpoints/baseline-best.ckpt \
        --image_a path/to/before.png \
        --image_b path/to/after.png \
        --output_dir outputs/inference_results
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.siam_unet import SiamUNet
from src.models.transformer_cd import TransformerCD
from src.models.cdformer import CDFormer
from src.inference.predictor import CDPredictor


def main(args):
    # Auto-detect device
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    print(f"Device: {device}")

    # Load model — in_channels=4: RGB + Canny edge boundary map
    if args.model_type == "baseline":
        model = SiamUNet(in_channels=4, classes=1)
    elif args.model_type == "custom":
        model = CDFormer(in_channels=4, classes=1)
    else:
        model = TransformerCD(in_channels=4, classes=1)

    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        # Only keep keys that belong to the raw model (prefix "model.").
        # Lightning checkpoints also store metric states (train_f1, val_recall, etc.)
        # which are not part of SiamUNet/TransformerCD and would cause a crash.
        state_dict = {
            k[len("model."):]: v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)
        print(f"Loaded weights: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}. Using random weights.")

    predictor = CDPredictor(model, device=device, tile_size=args.tile_size, overlap=args.overlap)

    print(f"Running inference on: {args.image_a} vs {args.image_b}")
    mask_255, prob_map, stats = predictor.predict(
        args.image_a,
        args.image_b,
        threshold=args.threshold,
        use_change_gate=not args.disable_change_gate,
        min_component_area=args.min_component_area,
        align_images=not args.disable_alignment,
        verify_components=not args.disable_component_verification,
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    mask_path = os.path.join(args.output_dir, "change_mask.png")
    prob_path = os.path.join(args.output_dir, "probability_map.png")
    overlay_path = os.path.join(args.output_dir, "overlay.png")

    cv2.imwrite(mask_path, mask_255)
    print(f"Saved binary mask: {mask_path}")

    # Probability heatmap
    prob_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(prob_path, prob_colored)
    print(f"Saved probability heatmap: {prob_path}")

    # Overlay on after image
    imgB = cv2.imread(args.image_b)
    color_mask = np.zeros_like(imgB)
    color_mask[:, :, 2] = mask_255  # Red
    color_mask[:, :, 1] = mask_255 // 4  # Slight orange
    overlay = imgB.copy()
    idx = mask_255 > 0
    overlay[idx] = cv2.addWeighted(imgB[idx], 0.5, color_mask[idx], 0.5, 0)
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (100, 100, 255), 2)
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved overlay: {overlay_path}")

    # Print stats
    print(f"\n── Results ──")
    print(f"  Changed area: {stats['changed_area_percentage']}%")
    print(f"  Detected regions: {stats['region_count']}")

    if stats["region_count"] > 0:
        print(f"  Summary: Detected {stats['region_count']} probable new or modified building region(s).")
    else:
        print(f"  Summary: No significant building changes detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local inference on a before/after image pair.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["baseline", "advanced", "custom"], default="custom")
    parser.add_argument("--image_a", type=str, required=True, help="Path to before image")
    parser.add_argument("--image_b", type=str, required=True, help="Path to after image")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_results")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min_component_area", type=int, default=150)
    parser.add_argument("--disable_change_gate", action="store_true")
    parser.add_argument("--disable_alignment", action="store_true")
    parser.add_argument("--disable_component_verification", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, mps, or cuda")

    args = parser.parse_args()
    main(args)
