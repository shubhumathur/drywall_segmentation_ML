# predict_utils.py
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model & Prompt Utilities
# -----------------------------

def get_model_weight(dataset_name):
    if dataset_name.lower() == "drywall_join":
        return "models/clipseg_drywall_join.pth"
    elif dataset_name.lower() == "cracks":
        return "models/clipseg_cracks.pth"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_prompt(dataset_name):
    """Return a fixed prompt for each dataset for consistency in evaluation."""
    if dataset_name.lower() == "drywall_join":
        return "segment taping area"
    elif dataset_name.lower() == "cracks":
        return "segment crack"
    else:
        return "segment the region of interest"

# -----------------------------
# Core Prediction Function
# -----------------------------

def predict_image(image_path, dataset_name, output_folder="results", threshold=0.5, prompt=None):
    """Predict segmentation mask and save with dynamic path handling."""
    
    weight_path = get_model_weight(dataset_name)
    
    if prompt is None:
        prompt = get_prompt(dataset_name)
    
    # Load model & processor
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
    # Dynamic path handling like predict.py
    # -----------------------------
    image_path = Path(image_path).resolve()
    parts = list(image_path.parts)
    
    try:
        idx = parts.index("dataset_preprocessing")
    except ValueError:
        raise ValueError("Image path must include 'dataset_preprocessing' folder")
    
    raw_data_root = Path(*parts[:idx+2]).resolve()  # includes dataset_preprocessing/raw
    relative_path = image_path.parent.relative_to(raw_data_root)  # preserves folder structure
    
    # Build output folder path
    output_dir = Path(output_folder) / relative_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mask with prompt in filename
    prompt_filename = prompt.replace(" ", "_")
    output_file = output_dir / f"{image_path.stem}__{prompt_filename}.png"
    cv2.imwrite(str(output_file), mask_255)
    
    print(f"✅ Saved mask: {output_file} with prompt: \"{prompt}\" and threshold {threshold}")
    return output_file, mask_255
