import torch
import torch.nn.functional as F
import os
import yaml
import sys
import random
import numpy as np
from tqdm import tqdm
import logging
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # ✅ For mixed precision
from prompts import get_random_prompt  # <- new

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"🔒 Reproducibility seeds set to {SEED}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_preprocessing')))
from data_prep import SegmentationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

log_file = config["logging"]["log_file"]
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

def get_processor(pretrained_model_name: str):
    processor = CLIPSegProcessor.from_pretrained(pretrained_model_name)
    processor.image_processor.do_rescale = False
    return processor

def get_dataloader(dataset_name, base_folder, split, batch_size, shuffle=True):
    dataset = SegmentationDataset(
        base_folder=os.path.join(base_folder, dataset_name),
        split=split,
        transform=None
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_model(dataset_config):
    dataset_name = dataset_config["name"]
    save_path = os.path.join(os.path.dirname(__file__), dataset_config["save_path"])
    base_folder = os.path.abspath(os.path.join(os.path.dirname(config_path), config["dataset"]["processed_folder"]))
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    weight_decay = config["training"]["weight_decay"]
    pretrained_model_name = config["model"]["pretrained_model_name"]
    early_stop_patience = config["training"].get("early_stop_patience", 5)

    logger.info(f"🔧 Starting training for dataset: {dataset_name}")

    processor = get_processor(pretrained_model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(pretrained_model_name)
    model.to(device)

    train_loader = get_dataloader(dataset_name, base_folder, "train", batch_size)
    val_loader = get_dataloader(dataset_name, base_folder, "valid", batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"🧠 {dataset_name} Epoch {epoch+1}/{epochs}", leave=False)

        for images, masks, _ in pbar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).float()
            optimizer.zero_grad()

            prompt = get_random_prompt(dataset_name)  # <- use descriptive prompt
            with autocast():
                inputs = processor(text=[prompt] * images.size(0), images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits.unsqueeze(1) if outputs.logits.dim() == 3 else outputs.logits
                logits_upsampled = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                bce_loss = bce_criterion(logits_upsampled, masks)
                d_loss = dice_loss(logits_upsampled, masks)
                loss = bce_loss + d_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"📉 {dataset_name} Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                prompt = get_random_prompt(dataset_name)
                inputs = processor(text=[prompt] * images.size(0), images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits.unsqueeze(1) if outputs.logits.dim() == 3 else outputs.logits
                logits_upsampled = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                bce_loss = bce_criterion(logits_upsampled, masks)
                d_loss = dice_loss(logits_upsampled, masks)
                val_loss += (bce_loss + d_loss).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        logger.info(f"🧪 {dataset_name} Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Best model saved for {dataset_name} (Epoch {epoch+1}, Val Loss {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"🛑 Early stopping triggered after {early_stop_patience} epochs without improvement.")
                break

    logger.info(f"🏁 Training complete for {dataset_name}. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    for ds in config["datasets"]:
        train_model(ds)
