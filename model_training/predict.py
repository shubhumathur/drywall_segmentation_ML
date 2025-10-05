# predict.py
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import os
import argparse
import random
import argparse
from predict_utils import predict_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_weight(dataset_name):
    if dataset_name.lower() == "drywall_join":
        return "models/clipseg_drywall_join.pth"
    elif dataset_name.lower() == "cracks":
        return "models/clipseg_cracks.pth"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_prompt(dataset_name):
    """Return a descriptive prompt from the pool for each dataset."""
    if dataset_name.lower() == "drywall_join":
        prompt_pool = [
            "segment the drywall taping or joint area on the wall surface",
            "find and mask the taped joints between drywall sheets",
            "highlight the drywall seams or taping regions accurately",
            "detect and segment the drywall taping areas"
        ]
    elif dataset_name.lower() == "cracks":
        prompt_pool = [
            "segment visible cracks or defects on the wall surface",
            "find and mask wall crack regions clearly",
            "highlight damaged or cracked portions of the wall",
            "detect and segment surface cracks and imperfections"
        ]
    else:
        prompt_pool = ["segment the region of interest"]
    return random.choice(prompt_pool)

def predict_image(image_path, dataset_name, output_folder="results", threshold=0.5):
    weight_path = get_model_weight(dataset_name)
    prompt = get_prompt(dataset_name)

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor.image_processor.do_rescale = False
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image) / 255.0
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    inputs = processor(text=[prompt], images=image_tensor, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Generate mask
    pred_mask = torch.sigmoid(outputs.logits).cpu().numpy()[0, 0]
    orig_h, orig_w = image.size[::-1]
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_255 = (pred_mask > threshold).astype(np.uint8) * 255

    # -----------------------------
    # Dynamic relative path logic
    # -----------------------------
    image_path = Path(image_path).resolve()
    parts = list(image_path.parts)

    # Find index of 'dataset_preprocessing' folder
    try:
        idx = parts.index("dataset_preprocessing")
    except ValueError:
        raise ValueError("Image path must include 'dataset_preprocessing' folder")

    raw_data_root = Path(*parts[:idx+2]).resolve()  # includes 'dataset_preprocessing/raw'
    relative_path = image_path.parent.relative_to(raw_data_root)  # e.g., cracks/test/images

    # Build output folder path
    output_dir = Path(output_folder) / relative_path
    os.makedirs(output_dir, exist_ok=True)

    # Include prompt in filename (spaces -> underscores)
    prompt_filename = prompt.replace(" ", "_")
    output_file = output_dir / f"{image_path.stem}__{prompt_filename}.png"

    cv2.imwrite(str(output_file), mask_255)
    print(f"✅ Saved mask: {output_file} with prompt: \"{prompt}\" and threshold {threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict segmentation mask for an image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--output_folder', type=str, default="results", help="Base folder to save output masks")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for mask binarization")
    args = parser.parse_args()

    img_path_lower = args.image_path.lower()
    if "drywall_join" in img_path_lower:
        dataset_name = "drywall_join"
    elif "cracks" in img_path_lower:
        dataset_name = "cracks"
    else:
        raise ValueError("Cannot infer dataset from image path. Include 'drywall_join' or 'cracks' in path.")

    predict_image(args.image_path, dataset_name, args.output_folder, threshold=args.threshold)
