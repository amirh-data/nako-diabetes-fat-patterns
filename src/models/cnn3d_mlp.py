import torch
import torch.nn as nn


class CNN3DMLP(nn.Module):
    def __init__(self, in_channels=2, embedding_dim=128, num_classes=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
        )

        self.embedding = nn.Linear(128, embedding_dim)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb
