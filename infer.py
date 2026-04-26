"""
infer.py — Fire Classifier Inference on Custom Image Pairs
===========================================================
Runs the trained MidFusionNet classifier on arbitrary RGB+Thermal image pairs.
No labels required. Replicates exact training preprocessing from dataset.py.

Usage:
    python infer.py --rgb_dir custom_test/rgb \
                    --thermal_dir custom_test/thermal \
                    --checkpoint fire_classifier.pth \
                    [--visualize]

Output:
    - Prints per-image predictions to console
    - Saves results to inference_results.json
    - If --visualize: saves side-by-side image panels to inference_vis/
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import MidFusionNet

# ── Label mapping ────────────────────────────────────────────────────────────
IDX_TO_LABEL = {0: "NN", 1: "YN", 2: "YY"}
LABEL_DESCRIPTIONS = {
    "NN": "No fire",
    "YN": "Fire (no heavy smoke)",
    "YY": "Fire + heavy smoke"
}
FIRE_CLASSES = {1, 2}  # classes that trigger localization in Phase 2

# ── Preprocessing — must match dataset.py exactly ───────────────────────────
# Training used raw ToTensor() only (no Normalize). Do NOT add ImageNet stats.
RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),          # → [0, 1], shape [3, 256, 256]
])

THERMAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),          # → [0, 1], shape [1, 256, 256]
])


def load_model(checkpoint_path: str, device: torch.device) -> MidFusionNet:
    model = MidFusionNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[✓] Model loaded from: {checkpoint_path}")
    return model


def get_image_pairs(rgb_dir: str, thermal_dir: str) -> list[str]:
    """Returns sorted list of filenames present in both modality folders."""
    rgb_files = set(os.listdir(rgb_dir))
    th_files = set(os.listdir(thermal_dir))
    paired = sorted(rgb_files & th_files)

    if not paired:
        raise FileNotFoundError(
            "No matching filenames found between rgb_dir and thermal_dir. "
            "Ensure RGB and thermal images share identical filenames."
        )

    unpaired = (rgb_files | th_files) - (rgb_files & th_files)
    if unpaired:
        print(f"[!] Warning: {len(unpaired)} files have no pair and will be skipped: {unpaired}")

    return paired


def run_inference(model, rgb_path, th_path, device):
    """
    Runs a single forward pass.
    Returns: predicted class index (int), softmax probabilities (np.array)
    """
    rgb = Image.open(rgb_path).convert("RGB")
    th = Image.open(th_path).convert("L")

    rgb_tensor = RGB_TRANSFORM(rgb).unsqueeze(0).to(device)   # [1, 3, 256, 256]
    th_tensor = THERMAL_TRANSFORM(th).unsqueeze(0).to(device) # [1, 1, 256, 256]

    with torch.no_grad():
        logits = model(rgb_tensor, th_tensor)                  # [1, 3]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [3]
        pred_idx = int(np.argmax(probs))

    return pred_idx, probs


def visualize_prediction(rgb_path, th_path, pred_label, probs, out_dir, filename):
    """Saves a 3-panel visualization: RGB | Thermal | Confidence bar chart."""
    os.makedirs(out_dir, exist_ok=True)

    rgb_img = Image.open(rgb_path).convert("RGB")
    th_img = Image.open(th_path).convert("L")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"{filename}  →  Prediction: {pred_label} ({LABEL_DESCRIPTIONS[pred_label]})",
        fontsize=12, fontweight="bold"
    )

    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(th_img, cmap="inferno")
    axes[1].set_title("Thermal")
    axes[1].axis("off")

    # Confidence bar chart
    colors = ["#4CAF50" if i == list(IDX_TO_LABEL.values()).index(pred_label)
              else "#90A4AE" for i in range(3)]
    bars = axes[2].bar(list(IDX_TO_LABEL.values()), probs, color=colors)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Class Probabilities")
    axes[2].set_ylabel("Softmax Probability")
    for bar, prob in zip(bars, probs):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{prob:.2f}", ha="center", fontsize=9
        )

    patch = mpatches.Patch(color="red" if pred_label in ("YN", "YY") else "green",
                           label="FIRE GATE: OPEN" if pred_label in ("YN", "YY") else "FIRE GATE: CLOSED")
    axes[2].legend(handles=[patch], loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_pred.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MidFusionNet Inference Script")
    parser.add_argument("--rgb_dir",     required=True, help="Path to RGB images folder")
    parser.add_argument("--thermal_dir", required=True, help="Path to thermal images folder")
    parser.add_argument("--checkpoint",  required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--output_json", default="inference_results.json",
                        help="Where to save structured results (default: inference_results.json)")
    parser.add_argument("--visualize",   action="store_true",
                        help="If set, saves visualization panels to inference_vis/")
    parser.add_argument("--vis_dir",     default="inference_vis",
                        help="Output folder for visualizations (default: inference_vis/)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Using device: {device}")

    model = load_model(args.checkpoint, device)
    pairs = get_image_pairs(args.rgb_dir, args.thermal_dir)
    print(f"[✓] Found {len(pairs)} image pairs\n")

    results = []
    fire_count = 0

    for filename in pairs:
        rgb_path = os.path.join(args.rgb_dir, filename)
        th_path  = os.path.join(args.thermal_dir, filename)

        pred_idx, probs = run_inference(model, rgb_path, th_path, device)
        pred_label = IDX_TO_LABEL[pred_idx]
        fire_detected = pred_idx in FIRE_CLASSES

        if fire_detected:
            fire_count += 1

        result = {
            "frame_id": os.path.splitext(filename)[0],
            "filename": filename,
            "classifier_prediction": pred_label,
            "fire_detected": fire_detected,
            "phase2_localization": fire_detected,   # gate flag for Phase 2
            "probabilities": {
                "NN": round(float(probs[0]), 4),
                "YN": round(float(probs[1]), 4),
                "YY": round(float(probs[2]), 4)
            },
            "bounding_boxes": []   # populated by Phase 2 localization
        }
        results.append(result)

        status = "🔥 FIRE" if fire_detected else "✓  CLEAR"
        print(f"  {status}  |  {filename:<30}  |  {pred_label}  |  "
              f"NN={probs[0]:.2f}  YN={probs[1]:.2f}  YY={probs[2]:.2f}")

        if args.visualize:
            visualize_prediction(rgb_path, th_path, pred_label, probs,
                                 args.vis_dir, filename)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "total_frames": len(pairs),
        "fire_detected": fire_count,
        "clear_frames": len(pairs) - fire_count,
        "fire_rate": round(fire_count / len(pairs), 4)
    }
    output = {"summary": summary, "results": results}

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[✓] Done. {fire_count}/{len(pairs)} frames flagged as fire.")
    print(f"[✓] Results saved to: {args.output_json}")
    if args.visualize:
        print(f"[✓] Visualizations saved to: {args.vis_dir}/")


if __name__ == "__main__":
    main()
