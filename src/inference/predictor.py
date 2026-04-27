import torch
import cv2
import numpy as np
from src.utils.tiler import Tiler
from src.data.transforms import get_inference_transforms
from src.data.dataset import _compute_edge_map

class CDPredictor:
    """
    End-to-End Inference pipeline for large satellite imagery using a PyTorch Model.
    Includes patching (tiling), batch inferencing, and morphological post-processing.

    Supports 4-channel model input (RGB + Canny edge boundary map) consistent with training.
    """
    def __init__(self, model, device="cpu", tile_size=256, overlap=64):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = get_inference_transforms()

    def _match_histograms(self, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Correct global brightness/contrast of src to match ref using mean-std normalization.
        Only fixes global lighting differences (different satellite pass, time of day).
        Unlike full histogram matching, this does NOT warp local colors, so real
        structural changes (new buildings) remain visible to the model.
        """
        matched = np.empty_like(src, dtype=np.float32)
        for c in range(src.shape[2]):
            s = src[:, :, c].astype(np.float32)
            r = ref[:, :, c].astype(np.float32)
            s_mean, s_std = s.mean(), s.std() + 1e-6
            r_mean, r_std = r.mean(), r.std() + 1e-6
            matched[:, :, c] = (s - s_mean) * (r_std / s_std) + r_mean
        return np.clip(matched, 0, 255).astype(np.uint8)

    def _build_suppression_mask(self, imgB_rgb):
        """
        Pixel-level weight map applied to the raw probability map before thresholding.

        Grey / achromatic pixels (bare earth, shadows, construction clearing) get a
        weight close to 0.05, so even a confident model score gets crushed below the
        detection threshold.  Coloured pixels (rooftops) keep weight 1.0.

        Formula: weight = clip((S - 10) / 50, 0.05, 1.0)
          S=10  (pure grey)       → weight = 0.00 → clipped to 0.05
          S=21  (light grey dirt) → weight = 0.22
          S=60  (muted colour)    → weight = 1.00
          S=120 (terracotta roof) → weight = 1.00
        """
        hsv = cv2.cvtColor(imgB_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1]                                    # 0–255

        weight = np.clip((saturation - 10.0) / 50.0, 0.05, 1.0)

        # Smooth edges so suppression doesn't create hard artefact lines
        weight = cv2.GaussianBlur(weight, (15, 15), 0)
        return weight

    def _build_change_gate(self, imgA_rgb, imgB_rgb, min_diff=0.08, softness=0.14):
        """
        Build a soft "actual visual change" gate from the before/after imagery.

        The model can sometimes fire on buildings that are present in both dates
        because they look building-like in image B. This gate keeps high model
        probabilities only where T1 and T2 differ after global brightness/contrast
        normalization. It is intentionally soft so subtle true additions are not
        erased before thresholding.
        """
        imgA_matched = self._match_histograms(imgA_rgb, imgB_rgb)

        color_delta = np.mean(
            np.abs(imgA_matched.astype(np.float32) - imgB_rgb.astype(np.float32)),
            axis=2,
        ) / 255.0

        gray_A = cv2.cvtColor(imgA_matched, cv2.COLOR_RGB2GRAY)
        gray_B = cv2.cvtColor(imgB_rgb, cv2.COLOR_RGB2GRAY)
        edge_A = _compute_edge_map(gray_A)
        edge_B = _compute_edge_map(gray_B)
        edge_delta = np.abs(edge_A - edge_B)

        color_delta = cv2.GaussianBlur(color_delta, (9, 9), 0)
        edge_delta = cv2.GaussianBlur(edge_delta, (9, 9), 0)
        combined = 0.75 * color_delta + 0.25 * edge_delta

        gate = np.clip((combined - min_diff) / softness, 0.05, 1.0)
        return cv2.GaussianBlur(gate.astype(np.float32), (11, 11), 0)

    def _build_change_score(self, imgA_rgb, imgB_rgb):
        """Raw before/after visual difference score used for component validation."""
        imgA_matched = self._match_histograms(imgA_rgb, imgB_rgb)
        color_delta = np.mean(
            np.abs(imgA_matched.astype(np.float32) - imgB_rgb.astype(np.float32)),
            axis=2,
        ) / 255.0

        gray_A = cv2.cvtColor(imgA_matched, cv2.COLOR_RGB2GRAY)
        gray_B = cv2.cvtColor(imgB_rgb, cv2.COLOR_RGB2GRAY)
        edge_delta = np.abs(_compute_edge_map(gray_A) - _compute_edge_map(gray_B))

        color_delta = cv2.GaussianBlur(color_delta, (9, 9), 0)
        edge_delta = cv2.GaussianBlur(edge_delta, (9, 9), 0)
        return 0.75 * color_delta + 0.25 * edge_delta

    def _verify_changed_components(self, mask_8u, imgA_rgb, imgB_rgb,
                                   min_mean_diff=0.045, min_p90_diff=0.09):
        """
        Remove components that look like stable buildings.

        Pixel probabilities can leak onto unchanged roofs. This pass evaluates
        each detected object as a whole and keeps it only if enough pixels inside
        the object show real before/after evidence.
        """
        imgA_matched = self._match_histograms(imgA_rgb, imgB_rgb)
        change_score = self._build_change_score(imgA_rgb, imgB_rgb)
        gray_A = cv2.cvtColor(imgA_matched, cv2.COLOR_RGB2GRAY)
        gray_B = cv2.cvtColor(imgB_rgb, cv2.COLOR_RGB2GRAY)
        edge_A = _compute_edge_map(gray_A)
        edge_B = _compute_edge_map(gray_B)
        edge_change = np.abs(edge_A - edge_B)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)
        verified = np.zeros_like(mask_8u)
        context_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        for i in range(1, num_labels):
            component = labels == i
            component_context = cv2.dilate(component.astype(np.uint8), context_kernel) > 0
            values = change_score[component_context]
            if values.size == 0:
                continue

            mean_diff = float(values.mean())
            p90_diff = float(np.percentile(values, 90))
            area = stats[i, cv2.CC_STAT_AREA]
            before_edge = float(edge_A[component_context].mean())
            after_edge = float(edge_B[component_context].mean())
            edge_delta = float(edge_change[component_context].mean())

            existed_before = before_edge >= 0.04
            similar_structure = abs(after_edge - before_edge) <= 0.02 and edge_delta <= 0.04

            # If the building existed before AND the structure looks the same,
            # reject it — same footprint = not a new building regardless of color shift.
            if existed_before and similar_structure:
                continue

            # Larger regions can be kept with slightly lower average change
            # because a new roof often includes mixed edges, shadows, and yards.
            large_component = area >= 900 and mean_diff >= 0.05 and p90_diff >= min_p90_diff
            clear_component = mean_diff >= min_mean_diff and p90_diff >= min_p90_diff
            structural_change = p90_diff >= 0.16 and edge_delta >= 0.05

            if clear_component or large_component or structural_change:
                verified[component] = 255

        return verified

    def _align_before_to_after(self, imgA_rgb, imgB_rgb):
        """
        Register T1 onto T2 with feature matching.

        The change mask is displayed over T2, so the before image must be warped
        into T2 coordinates. If matching is unreliable, fall back to resizing T1
        to T2 dimensions rather than inventing a bad transform.
        """
        if imgA_rgb.shape[:2] == imgB_rgb.shape[:2]:
            fallback = imgA_rgb
        else:
            fallback = cv2.resize(
                imgA_rgb,
                (imgB_rgb.shape[1], imgB_rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        gray_A = cv2.cvtColor(imgA_rgb, cv2.COLOR_RGB2GRAY)
        gray_B = cv2.cvtColor(imgB_rgb, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=5000)
        kp_A, desc_A = orb.detectAndCompute(gray_A, None)
        kp_B, desc_B = orb.detectAndCompute(gray_B, None)

        if desc_A is None or desc_B is None or len(kp_A) < 12 or len(kp_B) < 12:
            return fallback, False

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(desc_A, desc_B, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 12:
            return fallback, False

        src_pts = np.float32([kp_A[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_B[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99,
        )

        if matrix is None or inliers is None:
            return fallback, False

        inlier_count = int(inliers.sum())
        if inlier_count < 10 or inlier_count / max(len(good_matches), 1) < 0.25:
            return fallback, False

        aligned = cv2.warpAffine(
            imgA_rgb,
            matrix,
            (imgB_rgb.shape[1], imgB_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, True

    def post_process_mask(self, mask_binary, min_size=150, max_size=500000,
                          min_compactness=0.02, min_solidity=0.25):
        """
        Filters out false positive artifacts and closes holes in building footprints.

        Shape filters applied per connected component:
            Area        — min_size to max_size pixels
            Compactness — 4π·area / perimeter² (circle=1, spiky blob≈0)
                          Very low compactness → thin speckle or noise artifact
            Solidity    — area / convex_hull_area
                          Very low solidity → fragmented / non-building shape

        These complement the model-level suppression and eliminate speckle
        false positives that survive morphological cleanup.
        """
        mask_8u = (mask_binary * 255).astype(np.uint8)

        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(mask_8u,  cv2.MORPH_OPEN,  kernel_open)
        closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8)
        final_mask   = np.zeros_like(closed)
        region_count = 0
        total_pixels = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (min_size <= area <= max_size):
                continue

            component_mask = (labels == i).astype(np.uint8)

            # Compactness: 4π·A / P²
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perimeter = cv2.arcLength(contours[0], True)
            compactness = (4 * np.pi * area / (perimeter ** 2 + 1e-7))
            if compactness < min_compactness:
                continue

            # Solidity: area / convex hull area
            hull    = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-7)
            if solidity < min_solidity:
                continue

            final_mask[component_mask == 1] = 255
            region_count += 1
            total_pixels += area

        return final_mask, region_count, total_pixels

    def _run_single(self, tA: torch.Tensor, tB: torch.Tensor) -> np.ndarray:
        """Single forward pass → 2D probability map (H, W)."""
        logits = self.model(tA, tB)
        return torch.sigmoid(logits).squeeze().cpu().numpy()

    def _run_tta(self, tA: torch.Tensor, tB: torch.Tensor) -> np.ndarray:
        """
        Test-Time Augmentation for satellite imagery.

        Satellite images are captured top-down, so there is no privileged
        orientation — buildings look identical when flipped or rotated (D4
        symmetry). We run 4 variants and average their probability maps:
          1. Original
          2. Horizontal flip  → flip prediction back
          3. Vertical flip    → flip prediction back
          4. 90° CCW rotation → rotate prediction back

        Averaging reduces prediction variance and consistently improves F1
        by ~1–2% with zero additional training.
        """
        probs = []

        # 1 — original
        probs.append(self._run_single(tA, tB))

        # 2 — horizontal flip (last spatial dim = W)
        p = self._run_single(torch.flip(tA, dims=[-1]), torch.flip(tB, dims=[-1]))
        probs.append(np.flip(p, axis=-1).copy())

        # 3 — vertical flip (second-to-last spatial dim = H)
        p = self._run_single(torch.flip(tA, dims=[-2]), torch.flip(tB, dims=[-2]))
        probs.append(np.flip(p, axis=-2).copy())

        # 4 — 90° CCW rotation; undo with 90° CW (k=-1 in numpy = k=3 CCW)
        p = self._run_single(
            torch.rot90(tA, k=1, dims=[-2, -1]),
            torch.rot90(tB, k=1, dims=[-2, -1]),
        )
        probs.append(np.rot90(p, k=-1).copy())

        return np.mean(probs, axis=0)

    def predict(self, image_A_path, image_B_path, threshold=0.55, use_tta=False,
                use_suppression=False, use_change_gate=True, min_component_area=150,
                align_images=True, verify_components=True):
        imgA = cv2.imread(image_A_path)
        imgB = cv2.imread(image_B_path)

        if imgA is None or imgB is None:
            raise ValueError("Failed to load one or both images.")

        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        alignment_applied = False
        if align_images:
            imgA, alignment_applied = self._align_before_to_after(imgA, imgB)
        elif imgA.shape[:2] != imgB.shape[:2]:
            imgA = cv2.resize(imgA, (imgB.shape[1], imgB.shape[0]), interpolation=cv2.INTER_AREA)

        h, w, _ = imgB.shape
        tiler = Tiler((h, w), tile_size=self.tile_size, overlap=self.overlap)

        with torch.no_grad():
            for box in tiler.get_tiles_coords():
                patchA = tiler.crop(imgA, box)
                patchB = tiler.crop(imgB, box)

                # Compute Canny edge maps from raw patches (before normalization)
                # This mirrors how the dataset builds the 4th channel during training
                gray_A = cv2.cvtColor(patchA, cv2.COLOR_RGB2GRAY)
                gray_B = cv2.cvtColor(patchB, cv2.COLOR_RGB2GRAY)
                edge_A = _compute_edge_map(gray_A)  # (H, W) float32 [0, 1]
                edge_B = _compute_edge_map(gray_B)  # (H, W) float32 [0, 1]

                # Apply normalization to RGB patches
                transformed = self.transform(image=patchA, image0=patchB)
                tA = transformed['image']   # (3, H, W)
                tB = transformed['image0']  # (3, H, W)

                # Concatenate edge channel → (4, H, W) each
                edge_A_t = torch.tensor(edge_A).unsqueeze(0)  # (1, H, W)
                edge_B_t = torch.tensor(edge_B).unsqueeze(0)  # (1, H, W)
                tA_full = torch.cat([tA, edge_A_t], dim=0).unsqueeze(0).to(self.device)  # (1, 4, H, W)
                tB_full = torch.cat([tB, edge_B_t], dim=0).unsqueeze(0).to(self.device)  # (1, 4, H, W)

                if use_tta:
                    probs = self._run_tta(tA_full, tB_full)
                else:
                    probs = self._run_single(tA_full, tB_full)

                tiler.add_prediction(box, probs)

        # Reassemble full probability map
        stitched_probs = tiler.reassemble()

        # Change gate: if an object already exists in both images, it should not
        # be accepted purely because it resembles a building in image B.
        if use_change_gate:
            change_gate = self._build_change_gate(imgA, imgB)
            if change_gate.shape != stitched_probs.shape:
                change_gate = cv2.resize(change_gate,
                                         (stitched_probs.shape[1], stitched_probs.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            stitched_probs = stitched_probs * change_gate

        # Saturation-based suppression (OFF by default).
        # The old model needed this heuristic to suppress false positives on bare
        # earth and grey construction sites. CDFormer's Lovász loss, contrastive
        # regularization, and boundary supervision reduce those false positives at
        # the model level, and the morphological size filter below handles the rest.
        #
        # WARNING — only enable this if you are deploying in a region where there
        # are genuinely NO grey/concrete-roofed buildings. The formula crushes any
        # pixel with HSV saturation < ~60, which silently kills grey-roofed buildings
        # (common in Central Asia / Uzbekistan) even when the model is confident.
        if use_suppression:
            suppression = self._build_suppression_mask(imgB)
            if suppression.shape != stitched_probs.shape:
                suppression = cv2.resize(suppression,
                                         (stitched_probs.shape[1], stitched_probs.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            stitched_probs = stitched_probs * suppression

        binary_mask = (stitched_probs >= threshold).astype(np.float32)

        # Morphological cleanup + size filtering
        clean_mask, _, _ = self.post_process_mask(binary_mask, min_size=min_component_area)
        if verify_components:
            clean_mask = self._verify_changed_components(clean_mask, imgA, imgB)

        # Recount final regions for stats
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        region_count = num_labels - 1
        changed_pixels = int(np.sum(clean_mask > 0))

        total_img_pixels = h * w
        changed_area_percent = (changed_pixels / (total_img_pixels + 1e-7)) * 100

        stats = {
            "changed_area_percentage": round(changed_area_percent, 3),
            "region_count": region_count,
            "alignment_applied": alignment_applied
        }

        return clean_mask, stitched_probs, stats
