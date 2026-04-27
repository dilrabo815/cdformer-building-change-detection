"""
Dataset Preprocessing & Patch Tiling Script

Converts raw dataset downloads into the unified format:
    data/<dataset_name>/{train,val,test}/{A,B,label}/

Supports tiling large images into smaller patches (e.g., 1024x1024 -> 256x256).
Optionally discards empty patches (no change) with a controllable keep ratio.
Saves a CSV index for reproducibility.
"""

import os
import argparse
import csv
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


def tile_image(image, tile_size, overlap=0):
    """Split a single image into non-overlapping or overlapping patches."""
    h, w = image.shape[:2]
    stride = tile_size - overlap
    patches = []
    coords = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patch = image[y:y + tile_size, x:x + tile_size]
            patches.append(patch)
            coords.append((y, x))

    return patches, coords


def has_change(label_patch, threshold=10):
    """Returns True if the label patch has more than `threshold` changed pixels."""
    return np.sum(label_patch > 127) > threshold


def process_split(src_dir, dst_dir, split, tile_size, overlap, empty_keep_ratio, dataset_name):
    """Process a single split (train/val/test)."""
    src_A = os.path.join(src_dir, split, "A")
    src_B = os.path.join(src_dir, split, "B")
    src_label = os.path.join(src_dir, split, "label")

    if not os.path.isdir(src_A):
        print(f"  Skipping {split}: {src_A} does not exist.")
        return []

    dst_A = os.path.join(dst_dir, split, "A")
    dst_B = os.path.join(dst_dir, split, "B")
    dst_label = os.path.join(dst_dir, split, "label")
    os.makedirs(dst_A, exist_ok=True)
    os.makedirs(dst_B, exist_ok=True)
    os.makedirs(dst_label, exist_ok=True)

    image_files = sorted(glob(os.path.join(src_A, "*.png")) + glob(os.path.join(src_A, "*.jpg")))
    index_rows = []
    patch_id = 0

    for img_path in tqdm(image_files, desc=f"  {split}"):
        fname = os.path.basename(img_path)
        name_base = os.path.splitext(fname)[0]

        imgA = cv2.imread(os.path.join(src_A, fname))
        imgB_path = os.path.join(src_B, fname)
        label_path = os.path.join(src_label, fname)

        # Try alternate extension
        if not os.path.exists(imgB_path):
            imgB_path = os.path.join(src_B, fname.replace(".jpg", ".png"))
        if not os.path.exists(label_path):
            label_path = os.path.join(src_label, fname.replace(".jpg", ".png"))

        if imgA is None or not os.path.exists(imgB_path) or not os.path.exists(label_path):
            continue

        imgB = cv2.imread(imgB_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if imgA is None or imgB is None or label is None:
            continue

        h, w = imgA.shape[:2]

        # If image is already tile_size or smaller, just copy
        if h <= tile_size and w <= tile_size:
            out_name = f"{dataset_name}_{name_base}.png"
            cv2.imwrite(os.path.join(dst_A, out_name), imgA)
            cv2.imwrite(os.path.join(dst_B, out_name), imgB)
            cv2.imwrite(os.path.join(dst_label, out_name), label)
            change = has_change(label)
            index_rows.append([out_name, dataset_name, split, 0, 0, int(change)])
            continue

        # Tile
        patchesA, coords = tile_image(imgA, tile_size, overlap)
        patchesB, _ = tile_image(imgB, tile_size, overlap)
        patchesL, _ = tile_image(label, tile_size, overlap)

        for i, (pA, pB, pL, (cy, cx)) in enumerate(zip(patchesA, patchesB, patchesL, coords)):
            change = has_change(pL)

            # Optionally skip empty patches
            if not change and np.random.rand() > empty_keep_ratio:
                continue

            out_name = f"{dataset_name}_{name_base}_p{patch_id:05d}.png"
            cv2.imwrite(os.path.join(dst_A, out_name), pA)
            cv2.imwrite(os.path.join(dst_B, out_name), pB)
            cv2.imwrite(os.path.join(dst_label, out_name), pL)

            index_rows.append([out_name, dataset_name, split, cy, cx, int(change)])
            patch_id += 1

    return index_rows


def main(args):
    print(f"Processing dataset: {args.src_dir} -> {args.dst_dir}")
    print(f"  Tile size: {args.tile_size}, Overlap: {args.overlap}")
    print(f"  Empty patch keep ratio: {args.empty_keep_ratio}")

    dataset_name = os.path.basename(os.path.normpath(args.src_dir))
    all_rows = []

    for split in ["train", "val", "test"]:
        rows = process_split(
            args.src_dir, args.dst_dir, split,
            args.tile_size, args.overlap,
            args.empty_keep_ratio, dataset_name
        )
        all_rows.extend(rows)

    # Save CSV index
    csv_path = os.path.join(args.dst_dir, "patch_index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "source_dataset", "split", "offset_y", "offset_x", "has_change"])
        writer.writerows(all_rows)

    print(f"\nDone! {len(all_rows)} patches saved. Index: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tile a change detection dataset.")
    parser.add_argument("--src_dir", type=str, required=True, help="Path to raw dataset (must contain train/val/test with A/B/label)")
    parser.add_argument("--dst_dir", type=str, required=True, help="Output path for processed/tiled dataset")
    parser.add_argument("--tile_size", type=int, default=256, help="Patch size (default 256)")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between patches (default 0)")
    parser.add_argument("--empty_keep_ratio", type=float, default=0.1, help="Fraction of empty (no-change) patches to keep (default 0.1)")

    args = parser.parse_args()
    main(args)
