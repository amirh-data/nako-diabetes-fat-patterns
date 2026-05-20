import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def norm3d(num_channels, max_groups=8):
    """
    GroupNorm is more stable than BatchNorm3d for small 3D MRI batch sizes.
    Automatically finds a valid number of groups.
    """
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


# ---------------------------------------------------------------------------
# Model 1 – OptimizedCNN3D
# A plain 3-layer CNN with global average pooling.
# Fastest to train, good sanity-check baseline.
# ---------------------------------------------------------------------------

class OptimizedCNN3D(nn.Module):
    """
    Lightweight 3D CNN baseline.

    Input : (B, C, D, H, W), e.g. 96³ or 128³
    Output: logits (B, 2),  embedding (B, embedding_dim)
    """

    def __init__(self, in_channels, embedding_dim=128, num_classes=2):
        super().__init__()

        n_filters    = [32, 64, 128]
        dropout_rate = 0.20

        self.features = nn.Sequential(
            # block 1 – 96 → 48
            nn.Conv3d(in_channels, n_filters[0], kernel_size=3, padding=1, bias=False),
            norm3d(n_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # block 2 – 48 → 24
            nn.Conv3d(n_filters[0], n_filters[1], kernel_size=3, padding=1, bias=False),
            norm3d(n_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # block 3 – 24 → 12
            nn.Conv3d(n_filters[1], n_filters[2], kernel_size=3, padding=1, bias=False),
            norm3d(n_filters[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_filters[-1], embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x      = self.features(x)
        x      = self.pool(x)
        emb    = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb


# ---------------------------------------------------------------------------
# Model 2 & 3 – ResNet10_3D / ResNet18_3D
# Standard residual networks adapted for 3-D MRI (96^3 input).
# ---------------------------------------------------------------------------

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.norm2 = norm3d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                norm3d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ResNet3D(nn.Module):
    """
    Generic 3D ResNet backbone.

    Spatial flow for 96^3 input:
        stem      → 48^3
        maxpool   → 24^3
        layer2    → 12^3
        layer3    →  6^3
        layer4    →  3^3
        avgpool   →  1^3
    """

    def __init__(self, block, num_blocks, in_channels,
                 embedding_dim=128, num_classes=2, base_channels=32):
        super().__init__()

        self.in_planes = base_channels

        # stem: 96 → 48 → 24
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            norm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, base_channels,     num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, num_blocks[3], stride=2)

        final_channels = base_channels * 8 * block.expansion

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x      = self.stem(x)
        x      = self.layer1(x)
        x      = self.layer2(x)
        x      = self.layer3(x)
        x      = self.layer4(x)
        x      = self.pool(x)
        emb    = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb


def ResNet10_3D(in_channels, embedding_dim=128, num_classes=2):
    """
    Lightweight 3D ResNet (1 block per stage).
    ~3 M parameters with base_channels=32.
    Recommended first choice for ~3000 subjects.
    """
    return ResNet3D(
        block=BasicBlock3D,
        num_blocks=[1, 1, 1, 1],
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        base_channels=32,
    )


def ResNet18_3D(in_channels, embedding_dim=128, num_classes=2):
    """
    Larger 3D ResNet (2 blocks per stage).
    ~11 M parameters with base_channels=32.
    Try after ResNet10_3D once overfitting / memory are confirmed acceptable.
    """
    return ResNet3D(
        block=BasicBlock3D,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        base_channels=32,
    )


# ---------------------------------------------------------------------------
# Model 4 – ConvNeXt3D
# 3-D adaptation of ConvNeXt-Tiny (Liu et al., 2022).
# Uses depth-wise 7^3 convolutions + LayerNorm (channel-last style).
# More parameter-efficient than ResNet for the same receptive field.
# ---------------------------------------------------------------------------

class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt block adapted for 3-D volumes.

    Design:
        depthwise conv 7×7×7  (large kernel, same receptive field as 3 stacked 3×3)
        LayerNorm
        pointwise Linear × 2  (inverted bottleneck: expand 4×, then project back)
        GELU activation
        layer scale           (learnable per-channel scalar, initialised small)
    """

    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()

        # depthwise conv keeps channels separate → cheap large-kernel op
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)

        # inverted bottleneck in channel-last (linear) form
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # layer scale: initialise near zero so early training is stable
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        residual = x
        x = self.dwconv(x)

        # permute to channel-last for LayerNorm + Linear
        x = x.permute(0, 2, 3, 4, 1)          # (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)          # back to (B, C, D, H, W)

        return x + residual


class ConvNeXt3D(nn.Module):
    """
    3-D ConvNeXt (Tiny-scale) for MRI classification.

    Stage depths  : [3, 3, 9, 3]  (ConvNeXt-Tiny)
    Channel widths: [32, 64, 128, 256]  (halved vs 2-D Tiny to fit GPU)

    Spatial flow depends on input size; adaptive pooling allows 96³ or 128³.
        stem patchify  → 24^3  (stride-4 conv)
        downsample 2×  → 12^3
        downsample 2×  →  6^3
        downsample 2×  →  3^3
        adaptive pool  →  1^3

    Output: logits (B, 2),  embedding (B, embedding_dim)
    """

    def __init__(self, in_channels, embedding_dim=128, num_classes=2,
                 depths=(3, 3, 9, 3), dims=(32, 64, 128, 256)):
        super().__init__()

        # --- stem: non-overlapping patch embedding (stride 4 = 2×2 in each axis) ---
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4, bias=True),
            # LayerNorm in channel-last then back
            _LayerNorm3D(dims[0]),
        )

        # --- 4 stages with downsampling between them ---
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

            if i < 3:
                # 2× spatial downsampling + channel projection between stages
                self.downsamples.append(nn.Sequential(
                    _LayerNorm3D(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                ))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims[-1], embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.30),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.pool(x)                        # (B, C, 1, 1, 1)

        # LayerNorm on the pooled vector
        x = x.flatten(1)                        # (B, C)
        x = self.norm(x)

        emb    = self.embedding(x)
        logits = self.classifier(emb)

        return logits, emb


class _LayerNorm3D(nn.Module):
    """
    LayerNorm wrapper that accepts (B, C, D, H, W) tensors.
    Internally permutes to channel-last, applies LayerNorm, permutes back.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x
