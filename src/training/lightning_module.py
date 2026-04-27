import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall,
    MulticlassF1Score, MulticlassJaccardIndex,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """For boundary head — sparse 1-pixel ring targets where Lovász can be unstable."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter  = (probs * targets).sum()
        dice   = (2. * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovász-Hinge Loss.  Directly optimises IoU through a convex surrogate
    (Berman et al., CVPR 2018).  Consistently better than Dice under class imbalance.
    """
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.view(-1)
        targets = targets.view(-1).long()
        if targets.numel() == 0:
            return logits.sum() * 0.0
        signs  = 2.0 * targets.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = targets[perm]
        grad = _lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        """alpha=0.85 weights the positive (changed) class heavily → fewer missed buildings."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        at    = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (at * (1 - pt) ** self.gamma * bce).mean()


class FocalLovaszLoss(nn.Module):
    """Primary change detection loss: class-imbalance handling + direct IoU optimisation."""
    def __init__(self):
        super().__init__()
        self.focal  = FocalLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.lovasz(logits, targets)


# ---------------------------------------------------------------------------
# Object-level metric helper
# ---------------------------------------------------------------------------

def _object_level_metrics(probs: torch.Tensor, labels: torch.Tensor,
                           threshold: float = 0.5, iou_threshold: float = 0.5):
    """
    Object-level Precision / Recall / F1 via connected-component matching.

    Each predicted component is matched to the GT component with the highest IoU.
    A match counts if IoU >= iou_threshold.

    Slower than pixel metrics but reflects practical building-detection usefulness
    better: correctly localised buildings count as TP even if boundary pixels differ.

    Args:
        probs:  (N, 1, H, W) probability maps  (CPU float32)
        labels: (N, 1, H, W) binary GT masks   (CPU float32)
    Returns:
        (precision, recall, f1) floats
    """
    tp = fp = fn = 0
    for b in range(probs.shape[0]):
        pred_bin = (probs[b, 0].numpy() >= threshold).astype(np.uint8) * 255
        gt_bin   = (labels[b, 0].numpy() > 0.5).astype(np.uint8) * 255

        n_pred, pred_lbl, _, _ = cv2.connectedComponentsWithStats(
            pred_bin, connectivity=8)
        n_gt,   gt_lbl,   _, _ = cv2.connectedComponentsWithStats(
            gt_bin,   connectivity=8)

        matched_gt: set = set()
        for pi in range(1, n_pred):
            pm = pred_lbl == pi
            best_iou, best_gi = 0.0, -1
            for gi in range(1, n_gt):
                gm   = gt_lbl == gi
                inter = np.logical_and(pm, gm).sum()
                if inter == 0:
                    continue
                union = np.logical_or(pm, gm).sum()
                iou   = inter / (union + 1e-7)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn += max(0, (n_gt - 1) - len(matched_gt))

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)
    return float(precision), float(recall), float(f1)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class CDLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training Change Detection Models.

    Improvements over previous version
    -----------------------------------
    - Handles CDFormer 8-tuple: (logits, boundary, aux3, aux2, fA_feat, fB_feat,
                                  build_A_logits, build_B_logits)
    - Building-aware auxiliary losses (BCE + Dice on pseudo building masks)
    - Threshold calibration sweep at end of every validation epoch
      → logs val_best_threshold and val_best_f1_calibrated
    - Object-level P / R / F1 on up to 150 validation images per epoch
    - TorchMetrics BinaryF1/IoU accumulate TP/FP/FN across the full epoch
    - experiment_name and ablation_config saved into every checkpoint via save_hyperparameters
    """

    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4,
                 experiment_name: str = "experiment",
                 ablation_config: dict = None):
        super().__init__()
        self.model          = model
        self.learning_rate  = learning_rate
        self.weight_decay   = weight_decay
        self.experiment_name = experiment_name
        self.ablation_config = ablation_config or {}

        # Save all constructor args (except model weights) into Lightning checkpoints.
        # After training: torch.load(ckpt)['hyper_parameters'] shows the exact config.
        self.save_hyperparameters(ignore=['model'])

        self.criterion          = FocalLovaszLoss()
        self.boundary_criterion = DiceLoss()
        self.building_criterion = DiceLoss()      # for building aux heads

        # ── Per-class (changed) metrics ──────────────────────────────────────
        self.train_f1        = BinaryF1Score()
        self.val_f1          = BinaryF1Score()
        self.train_iou       = BinaryJaccardIndex()
        self.val_iou         = BinaryJaccardIndex()
        self.train_precision = BinaryPrecision()
        self.val_precision   = BinaryPrecision()
        self.train_recall    = BinaryRecall()
        self.val_recall      = BinaryRecall()

        # ── Macro metrics (matches published paper reporting style) ──────────
        self.train_mf1  = MulticlassF1Score(num_classes=2, average='macro')
        self.val_mf1    = MulticlassF1Score(num_classes=2, average='macro')
        self.train_miou = MulticlassJaccardIndex(num_classes=2, average='macro')
        self.val_miou   = MulticlassJaccardIndex(num_classes=2, average='macro')

        # ── Threshold calibration accumulators ───────────────────────────────
        self._thr_vals = np.arange(0.10, 0.92, 0.05).tolist()
        self._thr_tp   = {t: 0.0 for t in self._thr_vals}
        self._thr_fp   = {t: 0.0 for t in self._thr_vals}
        self._thr_fn   = {t: 0.0 for t in self._thr_vals}

        # ── Object-level metric sample buffers (capped to save memory) ───────
        self._val_probs:  list = []
        self._val_labels: list = []
        self._OBJ_CAP = 150   # max images stored per epoch for obj metrics

    # -------------------------------------------------------------------------
    def forward(self, x1, x2):
        return self.model(x1, x2)

    # -------------------------------------------------------------------------
    def _compute_loss(self, output, label, label_boundary, batch):
        """Extract all loss terms from model output tuple."""

        if isinstance(output, tuple) and len(output) == 8:
            # CDFormer 8-tuple — some elements may be None when ablation flags
            # disable the corresponding component (use_bgr=False → boundary_logits=None,
            # use_build_heads=False → build_A/B_logits=None).
            (logits, boundary_logits, aux3, aux2,
             fA_feat, fB_feat,
             build_A_logits, build_B_logits) = output

            label_s3 = F.interpolate(label, size=aux3.shape[2:], mode='nearest')
            label_s2 = F.interpolate(label, size=aux2.shape[2:], mode='nearest')

            # Main + deep supervision
            loss = (self.criterion(logits, label)
                    + 0.2 * self.criterion(aux3, label_s3)
                    + 0.1 * self.criterion(aux2, label_s2))

            # ── Boundary loss (skipped when use_bgr=False) ─────────────────
            if boundary_logits is not None:
                label_bnd_s = F.interpolate(label_boundary,
                                             size=boundary_logits.shape[2:], mode='nearest')
                loss = loss + 0.3 * self.boundary_criterion(boundary_logits, label_bnd_s)

            # ── Change Feature Contrastive Regularization ──────────────────
            cos_sim    = (fA_feat * fB_feat).sum(dim=1, keepdim=True)
            label_cs   = F.interpolate(label, size=cos_sim.shape[2:], mode='nearest')
            target_sim = 1.0 - 2.0 * label_cs                     # 0→+1, 1→−1
            loss = loss + 0.1 * F.mse_loss(cos_sim, target_sim)

            # ── Edge-Weighted Change Loss ───────────────────────────────────
            n_edge = label_boundary.sum()
            if n_edge > 0:
                edge_bce = F.binary_cross_entropy_with_logits(
                    logits * label_boundary, label * label_boundary,
                    reduction='sum') / n_edge
                loss = loss + 0.15 * edge_bce

            # ── Building-Aware Auxiliary Losses (skipped when use_build_heads=False)
            if (build_A_logits is not None
                    and 'label_build_A' in batch and 'label_build_B' in batch):
                lbA   = batch['label_build_A']
                lbB   = batch['label_build_B']
                lbA_s = F.interpolate(lbA, size=build_A_logits.shape[2:], mode='nearest')
                lbB_s = F.interpolate(lbB, size=build_B_logits.shape[2:], mode='nearest')
                build_loss = (0.5 * self.building_criterion(build_A_logits, lbA_s) +
                              0.5 * self.building_criterion(build_B_logits, lbB_s))
                loss = loss + 0.15 * build_loss

        elif isinstance(output, tuple) and len(output) == 4:
            # SiamUNet / TransformerCD legacy 4-tuple
            logits, boundary_logits, aux3, aux2 = output
            label_s3    = F.interpolate(label,          size=aux3.shape[2:],            mode='nearest')
            label_s2    = F.interpolate(label,          size=aux2.shape[2:],            mode='nearest')
            label_bnd_s = F.interpolate(label_boundary, size=boundary_logits.shape[2:], mode='nearest')
            loss = (self.criterion(logits, label)
                    + 0.3 * self.boundary_criterion(boundary_logits, label_bnd_s)
                    + 0.2 * self.criterion(aux3, label_s3)
                    + 0.1 * self.criterion(aux2, label_s2))

        elif isinstance(output, tuple):
            logits, boundary_logits = output
            loss = (self.criterion(logits, label)
                    + 0.3 * self.boundary_criterion(boundary_logits, label_boundary))
        else:
            logits = output
            loss   = self.criterion(logits, label)

        return loss, logits

    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        img_A          = batch['image_A']
        img_B          = batch['image_B']
        label          = batch['label']
        label_boundary = batch['label_boundary']

        output       = self(img_A, img_B)
        loss, logits = self._compute_loss(output, label, label_boundary, batch)

        probs       = torch.sigmoid(logits)
        targets_int = label.long()

        self.train_f1.update(probs.view(-1), targets_int.view(-1))
        self.train_iou.update(probs.view(-1), targets_int.view(-1))
        self.train_precision.update(probs.view(-1), targets_int.view(-1))
        self.train_recall.update(probs.view(-1), targets_int.view(-1))

        probs_2cls   = torch.cat([1.0 - probs, probs], dim=1)
        targets_2cls = targets_int.squeeze(1)
        self.train_mf1.update(probs_2cls, targets_2cls)
        self.train_miou.update(probs_2cls, targets_2cls)

        self.log('train_loss',      loss,                 on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train_f1',        self.train_f1,        on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mf1',       self.train_mf1,       on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou',       self.train_iou,       on_step=False, on_epoch=True)
        self.log('train_miou',      self.train_miou,      on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall',    self.train_recall,    on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # -------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        img_A          = batch['image_A']
        img_B          = batch['image_B']
        label          = batch['label']
        label_boundary = batch['label_boundary']

        output = self(img_A, img_B)

        # Eval mode → plain tensor; or 2-tuple for legacy models
        if isinstance(output, tuple):
            logits, boundary_logits = output[0], output[1]
            label_bnd_s = F.interpolate(label_boundary,
                                        size=boundary_logits.shape[2:], mode='nearest')
            loss = (self.criterion(logits, label)
                    + 0.3 * self.boundary_criterion(boundary_logits, label_bnd_s))
        else:
            logits = output
            loss   = self.criterion(logits, label)

        probs       = torch.sigmoid(logits)
        targets_int = label.long()

        self.val_f1.update(probs.view(-1), targets_int.view(-1))
        self.val_iou.update(probs.view(-1), targets_int.view(-1))
        self.val_precision.update(probs.view(-1), targets_int.view(-1))
        self.val_recall.update(probs.view(-1), targets_int.view(-1))

        probs_2cls   = torch.cat([1.0 - probs, probs], dim=1)
        targets_2cls = targets_int.squeeze(1)
        self.val_mf1.update(probs_2cls, targets_2cls)
        self.val_miou.update(probs_2cls, targets_2cls)

        # ── Threshold calibration: accumulate TP/FP/FN at each threshold ────
        probs_cpu  = probs.detach().cpu()
        labels_cpu = label.detach().cpu()
        for thr in self._thr_vals:
            preds = (probs_cpu >= thr).float()
            self._thr_tp[thr] += float((preds * labels_cpu).sum())
            self._thr_fp[thr] += float((preds * (1 - labels_cpu)).sum())
            self._thr_fn[thr] += float(((1 - preds) * labels_cpu).sum())

        # ── Collect samples for object-level metrics (memory-capped) ─────────
        if len(self._val_probs) < self._OBJ_CAP:
            self._val_probs.append(probs_cpu)
            self._val_labels.append(labels_cpu)

        self.log('val_loss',      loss,               prog_bar=True, sync_dist=True)
        self.log('val_f1',        self.val_f1,        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mf1',       self.val_mf1,       on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou',       self.val_iou,       on_step=False, on_epoch=True,                sync_dist=True)
        self.log('val_miou',      self.val_miou,      on_step=False, on_epoch=True,                sync_dist=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True,                sync_dist=True)
        self.log('val_recall',    self.val_recall,    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # -------------------------------------------------------------------------
    def on_validation_epoch_end(self):
        # ── Threshold calibration ─────────────────────────────────────────────
        best_f1, best_thr = 0.0, 0.5
        for thr in self._thr_vals:
            tp = self._thr_tp[thr]
            fp = self._thr_fp[thr]
            fn = self._thr_fn[thr]
            prec = tp / (tp + fp + 1e-7)
            rec  = tp / (tp + fn + 1e-7)
            f1   = 2 * prec * rec / (prec + rec + 1e-7)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        self.log('val_best_threshold',     float(best_thr), on_epoch=True, prog_bar=False)
        self.log('val_best_f1_calibrated', float(best_f1),  on_epoch=True, prog_bar=True)

        # Reset accumulators
        for thr in self._thr_vals:
            self._thr_tp[thr] = self._thr_fp[thr] = self._thr_fn[thr] = 0.0

        # ── Object-level metrics (on buffered validation samples) ─────────────
        if self._val_probs:
            all_probs  = torch.cat(self._val_probs,  dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)

            obj_p, obj_r, obj_f1 = _object_level_metrics(
                all_probs, all_labels, threshold=best_thr)

            self.log('val_obj_f1',        float(obj_f1), on_epoch=True, prog_bar=True)
            self.log('val_obj_precision',  float(obj_p),  on_epoch=True)
            self.log('val_obj_recall',     float(obj_r),  on_epoch=True)

        # Clear buffers
        self._val_probs.clear()
        self._val_labels.clear()

    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_epochs  = self.trainer.max_epochs if self.trainer else 100
        warmup_epochs = min(5, max(1, total_epochs // 20))
        cosine_epochs = max(1, total_epochs - warmup_epochs)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
