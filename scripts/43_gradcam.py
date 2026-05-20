"""
scripts/43_gradcam.py

Grad-CAM visualization for ResNet10_3D T2D classifier.

Generates 3D activation maps showing which MRI regions
the model focuses on when classifying T2D vs normal.

Saves:
    - 2D slices of Grad-CAM overlaid on fat/water MRI
    - Axial, coronal, sagittal views
    - Averaged maps for T2D and normal groups separately

Usage:
    python -u -m scripts.43_gradcam \
        --model_dir /path/to/model/dir \
        --n_subjects 20 \
        --input_mode fat_water_seg_trunk_binary
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from src.models.supervised_3d_models import ResNet10_3D
from src.data.nifti_dataset import CachedNiftiDataset, load_nifti_dataset

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")

MATCHED_CSV  = BASE / "cohorts/nako_t2d_normal_matched_cohort.csv"
CACHE_DIR    = BASE / "cached_nifti128/matched_fat_water_trunkseg"

MODEL_DIR    = BASE / "supervised_image_results/nifti128_cached_resnet10_3d_fat_water_seg_trunk_binary_matched_features-none"

OUT_DIR      = BASE / "gradcam_results"

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",   type=str,
                    default=str(MODEL_DIR))
parser.add_argument("--n_subjects",  type=int, default=20,
                    help="Number of subjects per class for Grad-CAM")
parser.add_argument("--input_mode",  type=str,
                    default="fat_water_seg_trunk_binary")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--target_layer",type=str, default="layer4",
                    help="Target layer for Grad-CAM hooks")
args = parser.parse_args()

MODEL_DIR = Path(args.model_dir)
OUT_DIR   = BASE / "gradcam_results" / MODEL_DIR.name
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Grad-CAM implementation for 3D CNN
# ---------------------------------------------------------------------

class GradCAM3D:
    """
    Grad-CAM for 3D CNN.
    Computes gradient-weighted class activation maps.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer = getattr(self.model, self.target_layer)
        self._hooks.append(
            layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def __call__(self, x, class_idx=None):
        """
        Compute Grad-CAM for input x.

        Args:
            x:          input tensor [1, C, D, H, W]
            class_idx:  target class (None = use predicted class)

        Returns:
            cam:        3D activation map [D, H, W] normalized 0-1
            pred_class: predicted class index
            pred_prob:  predicted probability for target class
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output, _ = self.model(x)
        pred_prob  = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        pred_class = class_idx

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Grad-CAM computation
        # gradients: [1, C, D, H, W]
        # activations: [1, C, D, H, W]
        gradients   = self.gradients[0]   # [C, D, H, W]
        activations = self.activations[0]  # [C, D, H, W]

        # Global average pool gradients over spatial dims
        weights = gradients.mean(dim=(1, 2, 3))  # [C]

        # Weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:],
                          device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = F.relu(cam)

        # Upsample to input size
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        cam = F.interpolate(cam,
                            size=x.shape[2:],
                            mode="trilinear",
                            align_corners=False)
        cam = cam.squeeze()  # [D, H, W]

        # Normalize to 0-1
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return (
            cam.cpu().numpy(),
            pred_class,
            pred_prob[0, class_idx].item(),
        )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class SimpleDataset(torch.utils.data.Dataset):
    """Load cached .pt files for Grad-CAM."""

    CHANNEL_MAP = {
        "fat":                        slice(0, 1),
        "fat_water":                  slice(0, 2),
        "seg_trunk_binary":           slice(2, 3),
        "seg_trunk_label":            slice(3, 4),
        "fat_water_seg_trunk_binary": [0, 1, 2],
        "fat_water_seg_trunk_label":  [0, 1, 3],
    }

    def __init__(self, df, cache_dir, input_mode):
        self.df         = df.reset_index(drop=True)
        self.cache_dir  = Path(cache_dir)
        self.input_mode = input_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        nako_id = int(row["NAKO_ID"])
        label   = int(row["label"])

        path   = self.cache_dir / f"NAKO751_{nako_id}.pt"
        sample = torch.load(path, map_location="cpu", weights_only=False)
        x      = sample["x"].float()

        ch = self.CHANNEL_MAP[self.input_mode]
        x  = x[ch] if isinstance(ch, list) else x[ch]

        return x, label, nako_id


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

def overlay_cam_on_slice(image_slice, cam_slice, alpha=0.5):
    """Overlay Grad-CAM heatmap on grayscale image slice."""
    # Normalize image to 0-1
    img = image_slice.copy()
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    # Convert to RGB
    img_rgb = np.stack([img, img, img], axis=-1)

    # Colormap for CAM
    heatmap = cm.jet(cam_slice)[:, :, :3]

    # Blend
    overlay = (1 - alpha) * img_rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def save_gradcam_figure(image, cam, nako_id, label, pred_class,
                        pred_prob, out_path):
    """
    Save Grad-CAM visualization with axial, coronal, sagittal views.
    image: [C, D, H, W] or [D, H, W]
    cam:   [D, H, W]
    """
    # Use fat channel (channel 0) for visualization
    if image.ndim == 4:
        fat = image[0]
    else:
        fat = image

    D, H, W = fat.shape

    # Middle slices
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    label_str = "T2D" if label == 1 else "Normal"
    pred_str  = "T2D" if pred_class == 1 else "Normal"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Grad-CAM | NAKO_ID={nako_id} | True={label_str} | "
        f"Pred={pred_str} (p={pred_prob:.3f})",
        fontsize=14,
    )

    views = [
        ("Axial (mid)",    fat[mid_d, :, :],  cam[mid_d, :, :]),
        ("Coronal (mid)",  fat[:, mid_h, :],  cam[:, mid_h, :]),
        ("Sagittal (mid)", fat[:, :, mid_w],  cam[:, :, mid_w]),
    ]

    for col, (title, img_slice, cam_slice) in enumerate(views):
        # Top row: fat image only
        axes[0, col].imshow(img_slice, cmap="gray", origin="lower")
        axes[0, col].set_title(f"{title} - Fat image")
        axes[0, col].axis("off")

        # Bottom row: Grad-CAM overlay
        overlay = overlay_cam_on_slice(img_slice, cam_slice)
        axes[1, col].imshow(overlay, origin="lower")
        axes[1, col].set_title(f"{title} - Grad-CAM overlay")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_average_cam_figure(avg_cam, label_str, out_path):
    """Save average Grad-CAM across a group."""
    D, H, W = avg_cam.shape
    mid_d   = D // 2
    mid_h   = H // 2
    mid_w   = W // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Average Grad-CAM — {label_str} group", fontsize=14)

    for ax, (title, cam_slice) in zip(axes, [
        ("Axial (mid)",    avg_cam[mid_d, :, :]),
        ("Coronal (mid)",  avg_cam[:, mid_h, :]),
        ("Sagittal (mid)", avg_cam[:, :, mid_w]),
    ]):
        im = ax.imshow(cam_slice, cmap="jet", origin="lower",
                       vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    print(f"Model dir:  {MODEL_DIR}")
    print(f"Output dir: {OUT_DIR}")

    # Check checkpoint
    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = MODEL_DIR / "final_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {MODEL_DIR}")
    print(f"Checkpoint: {ckpt_path}")

    # Load model
    in_channels = {
        "fat": 1, "fat_water": 2,
        "seg_trunk_binary": 1, "seg_trunk_label": 1,
        "fat_water_seg_trunk_binary": 3,
        "fat_water_seg_trunk_label":  3,
    }[args.input_mode]

    model = ResNet10_3D(in_channels=in_channels, embedding_dim=128)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: ResNet10_3D, in_channels={in_channels}")

    # Load matched cohort
    matched_df = pd.read_csv(MATCHED_CSV)
    matched_df = matched_df[
        matched_df["NAKO_ID"].apply(
            lambda x: (CACHE_DIR / f"NAKO751_{int(x)}.pt").exists()
        )
    ].copy().reset_index(drop=True)

    print(f"Available subjects: {len(matched_df)}")
    print(matched_df["label"].value_counts())

    # Sample subjects — equal T2D and normal
    n = args.n_subjects
    t2d_df    = matched_df[matched_df["label"] == 1].sample(
        n=min(n, len(matched_df[matched_df["label"] == 1])),
        random_state=42
    )
    normal_df = matched_df[matched_df["label"] == 0].sample(
        n=min(n, len(matched_df[matched_df["label"] == 0])),
        random_state=42
    )
    sample_df = pd.concat([t2d_df, normal_df]).reset_index(drop=True)

    print(f"Subjects for Grad-CAM: {len(sample_df)} "
          f"({len(t2d_df)} T2D, {len(normal_df)} normal)")

    dataset = SimpleDataset(sample_df, CACHE_DIR, args.input_mode)

    # Initialize Grad-CAM
    gradcam = GradCAM3D(model, target_layer=args.target_layer)

    # Store group-level CAMs
    t2d_cams    = []
    normal_cams = []

    results = []

    print("\nRunning Grad-CAM...", flush=True)

    for idx in range(len(dataset)):
        x, label, nako_id = dataset[idx]

        x_input = x.unsqueeze(0).to(device)
        x_input.requires_grad_(False)

        # Compute Grad-CAM for T2D class (class=1)
        cam, pred_class, pred_prob = gradcam(x_input, class_idx=1)

        print(
            f"[{idx+1}/{len(dataset)}] NAKO_ID={nako_id} | "
            f"True={'T2D' if label==1 else 'Normal'} | "
            f"Pred={'T2D' if pred_class==1 else 'Normal'} | "
            f"P(T2D)={pred_prob:.3f}",
            flush=True,
        )

        # Save individual figure
        fig_path = OUT_DIR / f"gradcam_{nako_id}_label{label}.png"
        save_gradcam_figure(
            x.numpy(), cam, nako_id, label,
            pred_class, pred_prob, fig_path
        )

        # Collect for group average
        if label == 1:
            t2d_cams.append(cam)
        else:
            normal_cams.append(cam)

        results.append({
            "NAKO_ID":    nako_id,
            "true_label": label,
            "pred_class": pred_class,
            "pred_prob":  pred_prob,
            "correct":    int(pred_class == label),
        })

    gradcam.remove_hooks()

    # Save average CAMs
    print("\nSaving average Grad-CAM maps...", flush=True)

    if t2d_cams:
        avg_t2d = np.mean(t2d_cams, axis=0)
        save_average_cam_figure(
            avg_t2d, "T2D",
            OUT_DIR / "gradcam_average_T2D.png"
        )
        np.save(OUT_DIR / "gradcam_average_T2D.npy", avg_t2d)

    if normal_cams:
        avg_normal = np.mean(normal_cams, axis=0)
        save_average_cam_figure(
            avg_normal, "Normal",
            OUT_DIR / "gradcam_average_Normal.png"
        )
        np.save(OUT_DIR / "gradcam_average_Normal.npy", avg_normal)

    # Difference map: T2D - Normal
    if t2d_cams and normal_cams:
        diff_cam = avg_t2d - avg_normal

        D, H, W  = diff_cam.shape
        mid_d    = D // 2
        mid_h    = H // 2
        mid_w    = W // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Grad-CAM Difference: T2D minus Normal\n"
                     "Red = more attention in T2D, Blue = more in Normal",
                     fontsize=13)

        vmax = np.abs(diff_cam).max()

        for ax, (title, diff_slice) in zip(axes, [
            ("Axial (mid)",    diff_cam[mid_d, :, :]),
            ("Coronal (mid)",  diff_cam[:, mid_h, :]),
            ("Sagittal (mid)", diff_cam[:, :, mid_w]),
        ]):
            im = ax.imshow(diff_slice, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(OUT_DIR / "gradcam_difference_T2D_vs_Normal.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        np.save(OUT_DIR / "gradcam_difference.npy", diff_cam)

    # Save results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "gradcam_results.csv", index=False)

    print(f"\nCorrect predictions: "
          f"{results_df['correct'].sum()}/{len(results_df)}")
    print(f"\nAll outputs saved to: {OUT_DIR}")

    # Summary
    print("\n" + "=" * 60)
    print("GRAD-CAM SUMMARY")
    print(f"Model:          ResNet10_3D")
    print(f"Input mode:     {args.input_mode}")
    print(f"Target layer:   {args.target_layer}")
    print(f"Subjects:       {len(results_df)} "
          f"({len(t2d_cams)} T2D, {len(normal_cams)} normal)")
    print(f"Accuracy:       "
          f"{results_df['correct'].mean():.3f}")
    print(f"Mean P(T2D) for T2D subjects:    "
          f"{results_df[results_df['true_label']==1]['pred_prob'].mean():.3f}")
    print(f"Mean P(T2D) for normal subjects: "
          f"{results_df[results_df['true_label']==0]['pred_prob'].mean():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()