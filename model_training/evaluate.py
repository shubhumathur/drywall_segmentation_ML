# evaluate.py (Evaluation-only version)
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# Add dataset_preprocessing to path for SegmentationDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_preprocessing')))
from data_prep import SegmentationDataset
from predict_utils import predict_image, get_prompt, get_model_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Metrics
# -------------------
def dice_score(pred_mask, true_mask, smooth=1e-6):
    pred = pred_mask.flatten()
    true = true_mask.flatten()
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum()
    return (2 * intersection + smooth) / (union + smooth)

def iou_score(pred_mask, true_mask, smooth=1e-6):
    pred = pred_mask.flatten()
    true = true_mask.flatten()
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# -------------------
# Evaluate dataset
# -------------------
def evaluate_dataset(dataset_name, base_folder, output_folder="results", threshold=0.5, num_examples=4):
    dataset = SegmentationDataset(
        base_folder=os.path.join(base_folder, dataset_name),
        split="test",
        transform=None
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_list, iou_list = [], []
    saved_examples = 0

    for idx, (image, mask, img_path, prompt) in enumerate(tqdm(loader, desc=f"Evaluating {dataset_name}")):
        image = image.to(device)
        mask = mask.numpy()[0,0]  # GT mask
        img_path = img_path[0]  # Since batch_size=1, img_path is a list
        prompt = prompt[0]  # Since batch_size=1, prompt is a list

        # Load predicted mask from output folder
        pred_mask_path = Path(output_folder) / dataset_name / "test" / "images" / (Path(img_path).stem + f"__{prompt.replace(' ', '_')}.png")
        if not pred_mask_path.exists():
            raise FileNotFoundError(f"Predicted mask not found: {pred_mask_path}")

        pred_mask = np.array(Image.open(pred_mask_path)) / 255.0

        dice_list.append(dice_score(pred_mask, mask))
        iou_list.append(iou_score(pred_mask, mask))

        # Save example visualization
        if saved_examples < num_examples:
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax[0].imshow(np.transpose(image.cpu().numpy()[0], (1,2,0)))
            ax[0].set_title("Original")
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("Ground Truth")
            ax[2].imshow(pred_mask, cmap="gray")
            ax[2].set_title("Prediction")
            for a in ax: a.axis("off")
            plt.tight_layout()
            vis_dir = Path(output_folder) / "visual_examples" / dataset_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(vis_dir / f"{Path(img_path).stem}_example.png")
            plt.close()
            saved_examples += 1

    print(f"--- {dataset_name} Metrics ---")
    print(f"Mean Dice: {np.mean(dice_list):.4f}")
    print(f"Mean IoU: {np.mean(iou_list):.4f}")

    return np.mean(dice_list), np.mean(iou_list)

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation on test dataset using existing predictions.")
    parser.add_argument('--datasets_folder', type=str, required=True, help="Path to dataset folder (e.g., dataset_preprocessing/data/cracks/test/images)")
    parser.add_argument('--output_folder', type=str, default="results", help="Folder containing predicted masks")
    parser.add_argument('--threshold', type=float, default=0.5, help="Mask threshold")
    args = parser.parse_args()

    datasets_folder = Path(args.datasets_folder)
    folder_str = str(datasets_folder).lower()
    if "drywall_join" in folder_str:
        dataset_name = "drywall_join"
    elif "cracks" in folder_str:
        dataset_name = "cracks"
    else:
        raise ValueError("Cannot infer dataset from folder path. Include 'drywall_join' or 'cracks' in path.")

    # Base folder for dataset is dataset_preprocessing/data
    base_folder = os.path.join(os.path.dirname(__file__), '..', 'dataset_preprocessing', 'data')

    evaluate_dataset(dataset_name, base_folder, args.output_folder, args.threshold)
