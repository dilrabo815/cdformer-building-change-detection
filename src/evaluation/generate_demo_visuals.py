import os
import argparse
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.models.siam_unet import SiamUNet
from src.models.transformer_cd import TransformerCD
from src.training.lightning_module import CDLightningModule
from src.data.dataset import ChangeDetectionDataset
from src.data.transforms import get_validation_transforms

def generate_overlay(img, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlays a binary mask onto an RGB image with transparency.
    Args:
        img: RGB image [H, W, 3] in [0, 255]
        mask: Binary mask [H, W] in [0, 1]
    """
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[mask > 0.5] = color
    
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Only keep overlay where mask is active
    result = np.where((mask > 0.5)[..., None], overlay, img)
    return result

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize base model — in_channels=4: RGB + Canny edge boundary map
    if args.model_type == 'baseline':
        base_model = SiamUNet(in_channels=4, classes=1)
    else:
        base_model = TransformerCD(in_channels=4, classes=1)
        
    model = CDLightningModule.load_from_checkpoint(args.checkpoint, model=base_model)
    model.to(device)
    model.eval()
    
    print(f"Generating Demo Visuals using model {args.checkpoint}")
    
    dataset = ChangeDetectionDataset(args.data_dir, subset="test", transform=get_validation_transforms(args.img_size))
    if len(dataset) == 0:
        dataset = ChangeDetectionDataset(args.data_dir, subset="val", transform=get_validation_transforms(args.img_size))
        
    if len(dataset) == 0:
        print(f"Error: No images found in {args.data_dir}")
        return

    # Pick N random samples
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            
            img_A_tensor = sample['image_A'].unsqueeze(0).to(device)
            img_B_tensor = sample['image_B'].unsqueeze(0).to(device)
            label_tensor = sample['label']
            filename = sample['filename']
            
            logits = model(img_A_tensor, img_B_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs > 0.5).astype(np.float32)
            
            # Map back to [0, 255] for plotting.
            # Slice [:3] to keep only RGB channels — the 4th channel is the edge map,
            # not needed for visualization.
            def to_numpy_img(tensor):
                arr = tensor.squeeze().cpu().numpy()
                arr = arr[:3, :, :]  # drop edge channel, keep RGB only
                arr = np.transpose(arr, (1, 2, 0))
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
                return (arr * 255).astype(np.uint8)

            img_A_np = to_numpy_img(sample['image_A'])
            img_B_np = to_numpy_img(sample['image_B'])
            gt_mask_np = label_tensor.squeeze().numpy()
            
            overlay_img = generate_overlay(img_B_np, pred_mask, color=(255, 50, 50))
            
            # Plot
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            
            axs[0].imshow(img_A_np)
            axs[0].set_title("Before (T1)")
            axs[0].axis('off')
            
            axs[1].imshow(img_B_np)
            axs[1].set_title("After (T2)")
            axs[1].axis('off')
            
            axs[2].imshow(gt_mask_np, cmap='gray')
            axs[2].set_title("Ground Truth")
            axs[2].axis('off')
            
            axs[3].imshow(pred_mask, cmap='gray')
            axs[3].set_title("Prediction")
            axs[3].axis('off')
            
            axs[4].imshow(overlay_img)
            axs[4].set_title("Overlay on T2")
            axs[4].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, f"demo_{filename.replace('.jpg','.png')}")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved demo visual: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate beautiful presentation-ready demo visuals.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .ckpt file")
    parser.add_argument("--model_type", type=str, choices=["baseline", "advanced"], default="baseline")
    parser.add_argument("--data_dir", type=str, default="data/ready/levir", help="Path to LEVIR-CD standardized directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to generate")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="outputs/demo_visuals")
    
    args = parser.parse_args()
    main(args)
