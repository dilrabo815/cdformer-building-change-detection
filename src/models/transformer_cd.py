import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AttentionFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x

class TransformerCD(nn.Module):
    """
    Advanced Model for Change Detection.
    Uses pretrained ResNet18 as the backbone for stable feature extraction,
    and Self-Attention modules to fuse the temporal difference effectively.

    Accepts 4-channel input (RGB + Canny edge boundary map).
    The pretrained ResNet18 first conv is patched to accept the 4th edge channel:
    the existing 3-channel weights are preserved and the 4th channel is initialized
    as the mean of the other three, which is the standard technique for adding
    input channels to pretrained models.

    Includes an auxiliary boundary head during training for precision/recall improvement.
    """
    def __init__(self, in_channels=4, classes=1):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Patch the first conv to accept in_channels (default 4: RGB + edge)
        # Preserve all pretrained weights for channels 0-2; init channel 3 as their mean.
        if in_channels != 3:
            old_conv = backbone.conv1  # (64, 3, 7, 7)
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                # Initialize extra channels as the mean of the 3 RGB channel weights
                for c in range(3, in_channels):
                    new_conv.weight[:, c, :, :] = old_conv.weight.mean(dim=1)
            backbone.conv1 = new_conv

        # Encoder Blocks
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)  # 64, H/4
        self.layer1 = backbone.layer1   # 64, H/4
        self.layer2 = backbone.layer2   # 128, H/8
        self.layer3 = backbone.layer3   # 256, H/16
        self.layer4 = backbone.layer4   # 512, H/32

        # Temporal Attention Fusion
        self.attn2 = AttentionFusion(128 * 2)
        self.attn3 = AttentionFusion(256 * 2)
        self.attn4 = AttentionFusion(512 * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512*2, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Sequential(nn.Conv2d(256 + 256*2, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(128 + 128*2, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.outc = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, classes, 1)
        )

        # Auxiliary boundary head — same role as in SiamUNet
        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, classes, kernel_size=1)
        )

    def extract_features(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return dict(f1=f1, f2=f2, f3=f3, f4=f4)

    def forward(self, x1, x2):
        feat1 = self.extract_features(x1)
        feat2 = self.extract_features(x2)

        # Concatenate temporal features
        f4_cat = torch.cat([feat1['f4'], feat2['f4']], dim=1)  # 1024
        f3_cat = torch.cat([feat1['f3'], feat2['f3']], dim=1)  # 512
        f2_cat = torch.cat([feat1['f2'], feat2['f2']], dim=1)  # 256

        # Temporal Attention Fusion
        f4_fused = self.attn4(f4_cat)
        f3_fused = self.attn3(f3_cat)
        f2_fused = self.attn2(f2_cat)

        # Decode
        x = self.up4(f4_fused)   # 256
        x = torch.cat([x, f3_fused], dim=1)
        x = self.conv4(x)

        x = self.up3(x)   # 128
        x = torch.cat([x, f2_fused], dim=1)
        x = self.conv3(x)

        x = self.up2(x)   # 64
        features = self.up1(x)   # 32 — shared features before final head

        logits = self.outc(features)

        if self.training:
            boundary_logits = self.boundary_head(features)
            return logits, boundary_logits

        return logits
