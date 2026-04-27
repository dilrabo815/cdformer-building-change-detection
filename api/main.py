from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
import cv2
import numpy as np
import torch
import sys
import base64
import json

# Ensure src/ is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.siam_unet import SiamUNet
from src.models.transformer_cd import TransformerCD
from src.models.cdformer import CDFormer
from src.inference.predictor import CDPredictor

app = FastAPI(title="Uzcosmos Building Change Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for our initialized predictor
predictor = None

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "demo_visuals")

def img_to_base64(img_array):
    """Encode an OpenCV image (BGR or grayscale) to base64 PNG string."""
    _, buffer = cv2.imencode(".png", img_array)
    return base64.b64encode(buffer).decode("utf-8")


@app.on_event("startup")
def load_model():
    global predictor
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing ML Engine on: {device}")
    
    # Try loading advanced model first, fallback to baseline
    # in_channels=4: RGB + Canny edge boundary map (matches training)
    model = SiamUNet(in_channels=4, classes=1)
    model_type = "baseline"

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    
    # Smarter Recursive Search for any .ckpt file
    import glob
    all_ckpts = glob.glob(os.path.join(checkpoint_dir, "**", "*.ckpt"), recursive=True)
    
    weight_path = None
    if all_ckpts:
        # Sort to find the most recent or highest epoch if multiple exist
        weight_path = sorted(all_ckpts, reverse=True)[0]
        
        # Determine model type from path or filename
        fname = os.path.basename(weight_path).lower()
        if "cdformer" in fname or "custom" in fname:
            model = CDFormer(in_channels=4, classes=1)
            model_type = "custom"
        elif "transformer" in fname or "advanced" in fname:
            model = TransformerCD(in_channels=4, classes=1)
            model_type = "advanced"
        else:
            model = SiamUNet(in_channels=4, classes=1)
            model_type = "baseline"
    else:
        # Fallback to defaults if no files found
        for candidate in ["advanced-best.ckpt", "baseline-best.ckpt"]:
            p = os.path.join(checkpoint_dir, candidate)
            if os.path.exists(p):
                weight_path = p
                if "advanced" in candidate:
                    model = TransformerCD(in_channels=4, classes=1)
                    model_type = "advanced"
                break

    if weight_path:
        try:
            state = torch.load(weight_path, map_location=device)
            # Only keep keys that belong to the raw model (prefix "model.").
            # Lightning checkpoints also store metric states (train_f1, val_recall, etc.)
            # which are not part of SiamUNet/TransformerCD and would cause a crash.
            state_dict = {
                k[len("model."):]: v
                for k, v in state["state_dict"].items()
                if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
            print(f"Loaded {model_type} weights from: {weight_path}")
        except Exception as e:
            print(f"Warning: Could not load weights ({e}). Running with untrained weights for demo.")
    else:
        print("No checkpoint found. Running with untrained weights for demonstration.")
        
    predictor = CDPredictor(model, device=device)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(predictor.device) if predictor else "uninitialized",
        "message": "Uzcosmos Change Detection API is running."
    }


@app.get("/samples")
def list_samples():
    """Return list of available demo sample images."""
    if not os.path.isdir(SAMPLES_DIR):
        return {"samples": [], "message": "No demo visuals found. Run generate_demo_visuals.py first."}
    
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith((".png", ".jpg"))]
    samples = []
    for f in sorted(files):
        filepath = os.path.join(SAMPLES_DIR, f)
        img = cv2.imread(filepath)
        if img is not None:
            samples.append({
                "filename": f,
                "image_base64": img_to_base64(img)
            })
    
    return {"samples": samples, "count": len(samples)}


@app.get("/metrics")
def get_metrics():
    """Return model configuration and capabilities info."""
    return {
        "model_info": {
            "baseline": "SiamUNet - Lightweight Siamese U-Net with shared encoder",
            "advanced": "TransformerCD - ResNet18 backbone with self-attention fusion"
        },
        "supported_formats": ["PNG", "JPG", "TIFF"],
        "tile_size": 256,
        "overlap": 64,
        "post_processing": {
            "morphological_open_kernel": 3,
            "morphological_close_kernel": 5,
            "min_component_area_px": 150,
            "change_gate": "enabled"
        },
        "datasets_supported": ["LEVIR-CD"]
    }


def build_overlay(image_bgr, mask_8u):
    """Draws a stylized glowing overlay on the image to show building changes."""
    color_mask = np.zeros_like(image_bgr)
    color_mask[:, :, 2] = mask_8u       # Red channel
    color_mask[:, :, 1] = mask_8u // 4   # Slight orange glow

    alpha = 0.5
    idx = (mask_8u > 0)

    overlay = image_bgr.copy()
    overlay[idx] = cv2.addWeighted(image_bgr[idx], 1 - alpha, color_mask[idx], alpha, 0)

    # Draw contours for glowing outlines
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (100, 100, 255), 2)

    return overlay


@app.post("/predict")
async def predict(file_A: UploadFile = File(...), file_B: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not initialized.")
        
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_A = os.path.join(tmpdir, "A_" + file_A.filename)
            path_B = os.path.join(tmpdir, "B_" + file_B.filename)
            
            with open(path_A, "wb") as f:
                f.write(await file_A.read())
            with open(path_B, "wb") as f:
                f.write(await file_B.read())
                
            # Run inference
            mask_255, prob_map, stats = predictor.predict(
                path_A,
                path_B,
                threshold=0.30,
                use_tta=True,
                use_change_gate=True,
                min_component_area=150,
                align_images=True,
                verify_components=True,
            )
            
            # Read images for overlay generation
            imgA = cv2.imread(path_A)
            imgB = cv2.imread(path_B)
            overlay = build_overlay(imgB, mask_255)
            
            # Generate probability heatmap (colorized)
            prob_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Build human-readable summary
            summary = "No significant building changes detected."
            if stats["region_count"] > 0:
                summary = f"Detected {stats['region_count']} probable new or modified building region(s) between the two timestamps, covering {stats['changed_area_percentage']}% of the total area."
            
            return JSONResponse(content={
                "status": "success",
                "stats": stats,
                "summary": summary,
                "mask_base64": img_to_base64(mask_255),
                "overlay_base64": img_to_base64(overlay),
                "heatmap_base64": img_to_base64(prob_colored),
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
