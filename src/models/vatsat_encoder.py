import torch
import torch.nn as nn
from src.models.supervised_3d_models import ResNet10_3D, ConvNeXt3D


class VATSATEncoder(nn.Module):
    """
    Wraps an existing backbone and adds a regression head
    for predicting VAT/SAT/tissue biomarkers.

    Returns: logits, embedding, regression_output
    """

    def __init__(self, backbone, num_targets=9):
        super().__init__()
        self.backbone   = backbone
        self.reg_head   = nn.Linear(128, num_targets)

    def forward(self, x):
        logits, emb = self.backbone(x)
        reg_out     = self.reg_head(emb)
        return logits, emb, reg_out


def build_vatsat_encoder(model_name, in_channels, num_targets=9):
    if model_name == "resnet10_3d":
        backbone = ResNet10_3D(in_channels=in_channels, embedding_dim=128)
    elif model_name == "convnext3d":
        backbone = ConvNeXt3D(in_channels=in_channels, embedding_dim=128)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return VATSATEncoder(backbone, num_targets=num_targets)
