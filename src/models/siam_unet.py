import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_new = torch.cat([avg_out, max_out], dim=1)
        x_new = self.conv1(x_new)
        return self.sigmoid(x_new)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SiamUNet(nn.Module):
    """
    Boundary-Enhanced Siamese U-Net for Change Detection.

    Encodes Image A and Image B (each with 4 channels: RGB + Canny edge map) using shared weights,
    concatenates feature maps along with their absolute difference in the decoder.

    Includes an auxiliary boundary prediction head trained with the GT label boundary,
    forcing the model to learn precise building outlines and greatly improving recall.
    The boundary head is only active during training (self.training = True).
    """
    def __init__(self, in_channels=4, classes=1):
        super().__init__()

        # Shared Encoder — accepts 4 channels (RGB + edge boundary map)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder (concatenates A, B, and abs(A-B) features along with upsampled features)
        # 1024(A) + 1024(B) + 1024(Diff) = 3072 -> Up to 512
        self.up1 = nn.ConvTranspose2d(3072, 512, kernel_size=2, stride=2)
        # Concat with up1(512) + 512(A) + 512(B) + 512(Diff) = 2048 -> 512
        self.conv1 = DoubleConv(2048, 512)
        self.cbam1 = CBAM(512)

        # up2: 512 -> 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Concat with up2(256) + 256(A) + 256(B) + 256(Diff) = 1024 -> 256
        self.conv2 = DoubleConv(1024, 256)
        self.cbam2 = CBAM(256)

        # up3: 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Concat with up3(128) + 128(A) + 128(B) + 128(Diff) = 512 -> 128
        self.conv3 = DoubleConv(512, 128)
        self.cbam3 = CBAM(128)

        # up4: 128 -> 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Concat with up4(64) + 64(A) + 64(B) + 64(Diff) = 256 -> 64
        self.conv4 = DoubleConv(256, 64)
        self.cbam4 = CBAM(64)

        # Main change detection output
        self.outc = nn.Conv2d(64, classes, kernel_size=1)

        # Auxiliary boundary head — teaches the model to predict building boundary rings.
        # Only used during training as an extra supervision signal.
        # At inference, this head is skipped so the API remains unchanged.
        self.boundary_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1)
        )

    def forward_encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def forward(self, x1, x2):
        # Shared weights encode both images (4-channel: RGB + edge map)
        a1, a2, a3, a4, a5 = self.forward_encoder(x1)
        b1, b2, b3, b4, b5 = self.forward_encoder(x2)

        # Decoder with absolute difference fusion at every level
        x = torch.cat([a5, b5, torch.abs(a5 - b5)], dim=1)

        x = self.up1(x)
        x = torch.cat([x, a4, b4, torch.abs(a4 - b4)], dim=1)
        x = self.conv1(x)
        x = self.cbam1(x)

        x = self.up2(x)
        x = torch.cat([x, a3, b3, torch.abs(a3 - b3)], dim=1)
        x = self.conv2(x)
        x = self.cbam2(x)

        x = self.up3(x)
        x = torch.cat([x, a2, b2, torch.abs(a2 - b2)], dim=1)
        x = self.conv3(x)
        x = self.cbam3(x)

        x = self.up4(x)
        x = torch.cat([x, a1, b1, torch.abs(a1 - b1)], dim=1)
        x = self.conv4(x)
        features = self.cbam4(x)   # (B, 64, H, W) — shared decoder features

        logits = self.outc(features)

        if self.training:
            # During training, also predict building boundaries for auxiliary supervision.
            # This forces the model to understand precise building shapes, not just rough blobs.
            boundary_logits = self.boundary_head(features)
            return logits, boundary_logits

        return logits
