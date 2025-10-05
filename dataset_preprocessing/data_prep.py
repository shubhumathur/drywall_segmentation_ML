# data_prep.py
import os
import shutil
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
import csv
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

# -----------------------------
# Helper: Aspect-ratio-preserving resize with padding
# -----------------------------
def resize_with_padding(image, desired_size=512):
    old_size = image.shape[:2]  # (h, w)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

# -----------------------------
# Step 4: Folder structure
# -----------------------------
def create_folder_structure(out_folder, dataset_name, splits=["train", "valid", "test"]):
    for split in splits:
        os.makedirs(os.path.join(out_folder, dataset_name, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, dataset_name, split, "masks"), exist_ok=True)

# -----------------------------
# Step 3 & 5: Process COCO
# -----------------------------
def process_coco_dataset(raw_folder, dataset_name, out_folder, img_size=512):
    splits = ["train", "valid", "test"]
    
    # --- Improved Prompt Engineering ---
    if dataset_name == "drywall_join":
        prompt_pool = [
            "segment the drywall taping or joint area on the wall surface",
            "find and mask the taped joints between drywall sheets",
            "highlight the drywall seams or taping regions accurately",
            "detect and segment the drywall taping areas"
        ]
    elif dataset_name == "cracks":
        prompt_pool = [
            "segment visible cracks or defects on the wall surface",
            "find and mask wall crack regions clearly",
            "highlight damaged or cracked portions of the wall",
            "detect and segment surface cracks and imperfections"
        ]
    else:
        prompt_pool = ["segment the region of interest"]

    # Randomly choose one prompt variant (adds natural language variation)
    prompt_string = random.choice(prompt_pool)
    
    
    for split in splits:
        images_dir = os.path.join(raw_folder, split)
        mask_dir = os.path.join(out_folder, dataset_name, split, "masks")
        img_out_dir = os.path.join(out_folder, dataset_name, split, "images")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(img_out_dir, exist_ok=True)
        
        coco_file = os.path.join(images_dir, "_annotations.coco.json")
        if not os.path.exists(coco_file):
            print(f"No COCO file in {images_dir}, skipping {split}")
            continue
        
        coco = COCO(coco_file)
        img_ids = list(coco.imgs.keys())
        
        for img_id in img_ids:
            img_info = coco.imgs[img_id]
            img_path = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                print(f"Image {img_path} not found, skipping")
                continue
            
            # Copy image to output folder
            shutil.copy(img_path, os.path.join(img_out_dir, img_info['file_name']))
            
            # Create blank mask
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                elif 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # Resize image and mask (aspect-ratio preserved)
            img_raw = cv2.imread(img_path)
            img_resized = resize_with_padding(img_raw, desired_size=img_size)
            mask_resized = resize_with_padding(mask, desired_size=img_size)
            mask_bin = (mask_resized > 0).astype(np.uint8) * 255  # enforce binary
            
            mask_name = os.path.splitext(img_info['file_name'])[0] + "_mask.png"
            cv2.imwrite(os.path.join(mask_dir, mask_name), mask_bin)
            cv2.imwrite(os.path.join(img_out_dir, img_info['file_name']), img_resized)
        
        print(f"✅ {dataset_name.upper()} | {split}: {len(img_ids)} images processed successfully.")

# -----------------------------
# Step 5.8: Prompts CSV
# -----------------------------
def create_prompts_csv(dataset_name, out_folder="dataset_preprocessing/data"):
    base = os.path.join(out_folder, dataset_name)
    prompt = "segment taping area" if dataset_name == "drywall_join" else "segment crack"
    csv_path = os.path.join(base, "prompts.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "mask_filename", "prompt"])
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(base, split, "images")
            mask_dir = os.path.join(base, split, "masks")
            if not os.path.exists(img_dir):
                continue
            for img_file in os.listdir(img_dir):
                mask_file = os.path.splitext(img_file)[0] + "_mask.png"
                writer.writerow([img_file, mask_file, prompt])
    print(f"✅ prompts.csv saved to {csv_path}")

# -----------------------------
# Step 6: PyTorch Dataset
# -----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, base_folder, split="train", transform=None, prompt=None):
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(base_folder, split, "images")
        self.masks_dir = os.path.join(base_folder, split, "masks")
        
        self.samples = []
        prompts_csv = os.path.join(base_folder, "prompts.csv")
        if os.path.exists(prompts_csv):
            prompt_map = {}
            with open(prompts_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt_map[row['image_filename']] = row['prompt']
            for img_file in os.listdir(self.images_dir):
                mask_file = os.path.splitext(img_file)[0] + "_mask.png"
                if mask_file in os.listdir(self.masks_dir):
                    self.samples.append((os.path.join(self.images_dir, img_file),
                                         os.path.join(self.masks_dir, mask_file),
                                         prompt_map.get(img_file, prompt)))
        else:
            for img_file in os.listdir(self.images_dir):
                mask_file = os.path.splitext(img_file)[0] + "_mask.png"
                if mask_file in os.listdir(self.masks_dir):
                    self.samples.append((os.path.join(self.images_dir, img_file),
                                         os.path.join(self.masks_dir, mask_file),
                                         prompt))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path, prompt = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = (augmented['mask'] > 0.5).astype(np.float32)
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, img_path, prompt

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['drywall_join', 'cracks'])
    parser.add_argument('--raw_folder', type=str, default='dataset_preprocessing/raw')
    parser.add_argument('--out_folder', type=str, default='dataset_preprocessing/data')
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()
    
    create_folder_structure(args.out_folder, args.dataset)
    process_coco_dataset(
        raw_folder=os.path.join(args.raw_folder, args.dataset),
        dataset_name=args.dataset,
        out_folder=args.out_folder,
        img_size=args.img_size
    )
    create_prompts_csv(args.dataset, out_folder=args.out_folder)
    print(f"🎯 {args.dataset} processing completed successfully!")
