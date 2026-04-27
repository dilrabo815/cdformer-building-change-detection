"""
CDFormer: Cross-Difference Transformer for Building Change Detection
Original architecture for the Uzcosmos project.

Key contributions
-----------------
1. Temporal Cross-Difference Module (TCDM)
2. Cross-Scale Change Propagation (CSCP)
3. Boundary-Guided Refinement (BGR)
4. Lightweight Global Context (LGC)  — single-layer MHSA at 1/32 scale
5. Building-Aware Auxiliary Heads    — per-timestep building presence
6. Multi-scale deep supervision
7. Change Feature Contrastive Regularization (in lightning_module.py)

Ablation flags
--------------
Each architectural component can be disabled independently for the ablation study:
    use_tcdm        False → replace TCDM with SimpleDiff (abs-diff + Conv)
    use_cscp        False → TCDMs receive no top-down context
    use_bgr         False → skip Boundary-Guided Refinement
    use_lgc         False → skip Lightweight Global Context
    use_build_heads False → skip per-timestep building aux heads

When a component is disabled its corresponding output in the training tuple is
set to None.  CDLightningModule._compute_loss checks for None before computing
the associated loss term so no code changes are needed in the training loop.

Training output (all flags enabled):
    (logits, boundary_logits, aux3, aux2, fA_feat, fB_feat,
     build_A_logits, build_B_logits)
    — components set to None when their flag is False.

Inference output: logits  (same regardless of flags)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnGelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Module 1 — Temporal Cross-Difference Module (TCDM)
# ---------------------------------------------------------------------------

class TemporalCrossDiffModule(nn.Module):
    """
    Bidirectional cross-channel attention gated by a spatial difference signal.
    O(C²) complexity.  Full description in original docstring at top of file.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(in_channels // 4, 16)

        self.diff_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.ch_attn = nn.Sequential(
            nn.Linear(in_channels * 2, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

    def forward(self, fA, fB, context=None):
        B, C, H, W = fA.shape
        diff_raw = torch.abs(fA - fB)
        if context is not None:
            diff_raw = diff_raw + context
        diff_feat = self.diff_enhance(diff_raw)

        gA = F.adaptive_avg_pool2d(fA, 1).view(B, C)
        gB = F.adaptive_avg_pool2d(fB, 1).view(B, C)
        w_AB = self.ch_attn(torch.cat([gA, gB], dim=1)).view(B, C, 1, 1)
        w_BA = self.ch_attn(torch.cat([gB, gA], dim=1)).view(B, C, 1, 1)

        cross_diff = fB * w_BA - fA * w_AB
        gate       = self.spatial_gate(diff_feat)
        change_feat = gate * diff_feat + (1.0 - gate) * cross_diff.abs()
        return self.out_proj(torch.cat([change_feat, cross_diff], dim=1))


# ---------------------------------------------------------------------------
# Ablation fallback — SimpleDiff (replaces TCDM when use_tcdm=False)
# ---------------------------------------------------------------------------

class SimpleDiff(nn.Module):
    """
    Ablation baseline: plain abs-difference projected through a ConvBnGelu.
    Same interface as TCDM (accepts context, returns same spatial shape).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = ConvBnGelu(in_channels, in_channels)

    def forward(self, fA, fB, context=None):
        diff = torch.abs(fA - fB)
        if context is not None:
            diff = diff + context
        return self.proj(diff)


# ---------------------------------------------------------------------------
# Module 2 — Boundary-Guided Refinement (BGR)
# ---------------------------------------------------------------------------

class BoundaryGuidedRefinement(nn.Module):
    """
    Predicts building boundaries and feeds the probability map back into the
    decoder feature stream to sharpen change predictions at building edges.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.boundary_pred = nn.Sequential(
            ConvBnGelu(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, 1, 1),
        )
        self.refine = ConvBnGelu(in_channels + 1, in_channels)

    def forward(self, feat, return_boundary=False):
        boundary_logits = self.boundary_pred(feat)
        boundary_prob   = torch.sigmoid(boundary_logits)
        refined = self.refine(torch.cat([feat, boundary_prob], dim=1))
        if return_boundary:
            return refined, boundary_logits
        return refined


# ---------------------------------------------------------------------------
# Module 3 — Lightweight Global Context (LGC)
# ---------------------------------------------------------------------------

class LightweightGlobalContext(nn.Module):
    """
    Single-layer pre-norm transformer encoder block (MHSA + FFN).
    Applied at the deepest encoder scale (1/32 → 64 tokens at 256-px input).
    Provides long-range context before the top-down decoder pass.
    """

    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn  = nn.MultiheadAttention(in_channels, num_heads,
                                            dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn   = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)           # B, HW, C
        n = self.norm1(tokens)
        attn_out, _ = self.attn(n, n, n)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# CDFormer — full model with ablation support
# ---------------------------------------------------------------------------

class CDFormer(nn.Module):
    """
    CDFormer with per-component ablation flags.

    Constructor flags (all default True = full model):
        use_tcdm        — bidirectional cross-channel attention change fusion
        use_cscp        — top-down context propagation between scales
        use_bgr         — boundary-guided decoder refinement
        use_lgc         — lightweight global context at 1/32 scale
        use_build_heads — per-timestep building presence aux heads

    Ablation presets for the paper's ablation table (run via train.py flags):
        Baseline SiamUNet          → --model baseline
        + BGR                      → --model custom --no_tcdm --no_cscp --no_lgc --no_build_heads
        + TCDM                     → --model custom --no_cscp --no_lgc --no_build_heads
        + CSCP                     → --model custom --no_lgc --no_build_heads
        + building aux heads       → --model custom --no_lgc
        + global context (LGC)     → --model custom          (full CDFormer)
        + hard negative mining     → --model custom --hard_negative_mining
        Full model                 → --model custom --hard_negative_mining  (same as above)
    """

    _FEAT_CH = [24, 40, 112, 320]
    _DEC_CH  = [128, 64, 32, 32]

    def __init__(self, in_channels: int = 4, classes: int = 1,
                 use_tcdm: bool = True, use_cscp: bool = True,
                 use_bgr: bool = True, use_lgc: bool = True,
                 use_build_heads: bool = True):
        super().__init__()

        self.use_tcdm        = use_tcdm
        self.use_cscp        = use_cscp
        self.use_bgr         = use_bgr
        self.use_lgc         = use_lgc
        self.use_build_heads = use_build_heads

        # ── Shared EfficientNet-B0 backbone ─────────────────────────────────
        backbone = timm.create_model(
            'efficientnet_b0', pretrained=True,
            features_only=True, out_indices=(1, 2, 3, 4),
        )
        if in_channels != 3:
            old = backbone.conv_stem
            new_conv = nn.Conv2d(in_channels, old.out_channels,
                                  kernel_size=old.kernel_size, stride=old.stride,
                                  padding=old.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old.weight.clone()
                for c in range(3, in_channels):
                    new_conv.weight[:, c] = old.weight.mean(dim=1)
            backbone.conv_stem = new_conv
        self.backbone = backbone

        fc, dc = self._FEAT_CH, self._DEC_CH

        # ── Change fusion modules (TCDM or SimpleDiff) ───────────────────────
        _DiffCls = TemporalCrossDiffModule if use_tcdm else SimpleDiff
        self.tcdm4 = _DiffCls(fc[3])
        self.tcdm3 = _DiffCls(fc[2])
        self.tcdm2 = _DiffCls(fc[1])
        self.tcdm1 = _DiffCls(fc[0])

        # ── Lightweight Global Context (optional) ─────────────────────────────
        if use_lgc:
            self.lgc = LightweightGlobalContext(fc[3], num_heads=4)

        # ── Cross-Scale Change Propagation context projections (optional) ─────
        if use_cscp:
            self.ctx43 = ConvBnGelu(fc[3], fc[2], k=1, p=0)
            self.ctx32 = ConvBnGelu(fc[2], fc[1], k=1, p=0)
            self.ctx21 = ConvBnGelu(fc[1], fc[0], k=1, p=0)

        # ── Decoder lateral projections ───────────────────────────────────────
        self.lat4 = ConvBnGelu(fc[3], dc[0], k=1, p=0)
        self.lat3 = ConvBnGelu(fc[2], dc[1], k=1, p=0)
        self.lat2 = ConvBnGelu(fc[1], dc[2], k=1, p=0)
        self.lat1 = ConvBnGelu(fc[0], dc[3], k=1, p=0)

        # ── Decoder blocks ────────────────────────────────────────────────────
        self.dec4 = ConvBnGelu(dc[0],           dc[0])
        self.dec3 = ConvBnGelu(dc[0] + dc[1],   dc[1])
        self.dec2 = ConvBnGelu(dc[1] + dc[2],   dc[2])
        self.dec1 = ConvBnGelu(dc[2] + dc[3],   dc[3])

        # ── Boundary-Guided Refinement (optional) ─────────────────────────────
        if use_bgr:
            self.bgr = BoundaryGuidedRefinement(dc[1])

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(ConvBnGelu(dc[3], 16), nn.Conv2d(16, classes, 1))

        # ── Deep supervision (always active — separate from ablation) ─────────
        self.aux_head3 = nn.Conv2d(dc[1], classes, 1)
        self.aux_head2 = nn.Conv2d(dc[2], classes, 1)

        # ── Per-timestep building aux heads (optional) ────────────────────────
        if use_build_heads:
            self.build_A_head = nn.Sequential(
                ConvBnGelu(fc[2], 32, k=1, p=0), nn.Conv2d(32, classes, 1))
            self.build_B_head = nn.Sequential(
                ConvBnGelu(fc[2], 32, k=1, p=0), nn.Conv2d(32, classes, 1))

    # -------------------------------------------------------------------------
    def _encode(self, x):
        return self.backbone(x)

    # -------------------------------------------------------------------------
    def forward(self, xA, xB):
        fA = self._encode(xA)
        fB = self._encode(xB)

        # ── Change fusion at deepest scale ────────────────────────────────────
        c4 = self.tcdm4(fA[3], fB[3])
        if self.use_lgc:
            c4 = self.lgc(c4)

        # ── Cross-Scale Change Propagation (top-down) or no context ──────────
        if self.use_cscp:
            ctx_3 = F.interpolate(self.ctx43(c4), size=fA[2].shape[2:],
                                   mode='bilinear', align_corners=False)
            c3 = self.tcdm3(fA[2], fB[2], context=ctx_3)

            ctx_2 = F.interpolate(self.ctx32(c3), size=fA[1].shape[2:],
                                   mode='bilinear', align_corners=False)
            c2 = self.tcdm2(fA[1], fB[1], context=ctx_2)

            ctx_1 = F.interpolate(self.ctx21(c2), size=fA[0].shape[2:],
                                   mode='bilinear', align_corners=False)
            c1 = self.tcdm1(fA[0], fB[0], context=ctx_1)
        else:
            c3 = self.tcdm3(fA[2], fB[2])
            c2 = self.tcdm2(fA[1], fB[1])
            c1 = self.tcdm1(fA[0], fB[0])

        # ── FPN decoder ───────────────────────────────────────────────────────
        x4    = self.dec4(self.lat4(c4))
        x4_up = F.interpolate(x4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x3    = self.dec3(torch.cat([x4_up, self.lat3(c3)], dim=1))

        # ── Boundary-Guided Refinement (optional) ─────────────────────────────
        boundary_logits = None
        if self.use_bgr:
            if self.training:
                x3, boundary_logits = self.bgr(x3, return_boundary=True)
            else:
                x3 = self.bgr(x3)

        x3_up = F.interpolate(x3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x2    = self.dec2(torch.cat([x3_up, self.lat2(c2)], dim=1))
        x2_up = F.interpolate(x2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x1    = self.dec1(torch.cat([x2_up, self.lat1(c1)], dim=1))

        x_out  = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        logits = self.head(x_out)

        if self.training:
            aux3 = self.aux_head3(x3)
            aux2 = self.aux_head2(x2)

            # Contrastive features (always returned — backbone is always active)
            fA_feat = F.normalize(fA[2], dim=1)
            fB_feat = F.normalize(fB[2], dim=1)

            # Per-timestep building heads (None if disabled)
            if self.use_build_heads:
                build_A_logits = self.build_A_head(fA[2])
                build_B_logits = self.build_B_head(fB[2])
            else:
                build_A_logits = None
                build_B_logits = None

            return (logits, boundary_logits, aux3, aux2,
                    fA_feat, fB_feat,
                    build_A_logits, build_B_logits)

        return logits
