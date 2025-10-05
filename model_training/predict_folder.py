# batch_predict.py
import argparse
from pathlib import Path
from predict import predict_image  # import the function from your predict.py

def batch_predict(input_folder, output_folder="results", threshold=0.5):
    input_folder = Path(input_folder).resolve()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # Recursively find all image files
    all_images = [p for p in input_folder.rglob("*") if p.suffix.lower() in image_extensions]

    if not all_images:
        print(f"No images found in {input_folder}")
        return

    for img_path in all_images:
        # Automatically infer dataset from path (same logic as predict.py)
        img_path_lower = str(img_path).lower()
        if "drywall_join" in img_path_lower:
            dataset_name = "drywall_join"
        elif "cracks" in img_path_lower:
            dataset_name = "cracks"
        else:
            print(f"Skipping {img_path} (cannot infer dataset)")
            continue

        predict_image(str(img_path), dataset_name, output_folder, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict segmentation masks for a folder.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to input images folder")
    parser.add_argument('--output_folder', type=str, default="results", help="Base folder to save output masks")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for mask binarization")
    args = parser.parse_args()

    batch_predict(args.input_folder, args.output_folder, args.threshold)
