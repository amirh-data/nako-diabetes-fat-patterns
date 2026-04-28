import torch
from src.models.cnn3d_mlp import CNN3DMLP


def main():
    model = CNN3DMLP()

    x = torch.randn(2, 2, 96, 96, 96)

    logits, emb = model(x)

    print("Input:", x.shape)
    print("Logits:", logits.shape)
    print("Embedding:", emb.shape)


if __name__ == "__main__":
    main()
