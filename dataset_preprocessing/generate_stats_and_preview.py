# generate_stats_and_preview.py
import os
import json
import numpy as np
import cv2
from PIL import Image

datasets = ["drywall_join", "cracks"]
base_folder = "dataset_preprocessing/data"
preview_folder = "dataset_preprocessing/results"
os.makedirs(preview_folder, exist_ok=True)

for dataset in datasets:
    stats = {}
    base = os.path.join(base_folder, dataset)
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(base, split, "images")
        mask_dir = os.path.join(base, split, "masks")
        if not os.path.exists(img_dir):
            continue
        files = os.listdir(img_dir)
        stats[split] = len(files)
        
        # Save 5 sample overlay images per split
        for i, img_file in enumerate(files[:5]):
            img_path = os.path.join(img_dir, img_file)
            mask_file = os.path.splitext(img_file)[0]+"_mask.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
            preview_path = os.path.join(preview_folder, f"{dataset}_{split}_{i+1}.png")
            cv2.imwrite(preview_path, overlay)
    
    # Save stats.json
    stats_path = os.path.join(base, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"{dataset}: stats.json saved with {stats} and previews generated.")
