# Drywall & Cracks Segmentation Project

## Project Overview

This project focuses on segmenting drywall taping areas and cracks on wall surfaces using deep learning. It is designed for construction AI applications to detect and highlight damaged or joint areas on drywall surfaces.

The project consists of dataset preparation, model training, prediction, and evaluation components.

---

## Folder Structure

```
drywall_segmentation/
├─ dataset_preprocessing/
│  ├─ raw/                     # Raw downloaded datasets (COCO format)
│  ├─ data/                    # Processed, train-ready datasets (images + binary masks)
│  ├─ generate_stats_and_preview.py  # Script to generate dataset stats and sanity check previews
│  ├─ data_prep.py             # Dataset processing script (COCO to masks, resizing, prompts)
│  ├─ test_dataset.py          # Quick PyTorch dataset loader test
├─ model_training/
│  ├─ train.py                 # Model training script
│  ├─ predict.py               # Single image prediction script
│  ├─ predict_folder.py        # Batch prediction script for folders
│  ├─ predict_utils.py         # Core prediction utilities and prompt management
│  ├─ evaluate.py              # Evaluation script for predicted masks
│  ├─ prompts.py               # Prompt templates (if any)
│  ├─ results/                 # Folder for saving prediction results and visualizations
├─ README.md                   # Project overview and instructions
├─ config.yaml                 # Configuration file (if applicable)
├─ requirements.txt (not present, see below)
```

---

## What the Project Does

- Downloads and prepares drywall join and cracks datasets from Roboflow in COCO format.
- Converts polygon and bounding box annotations into binary masks.
- Resizes images and masks to 512x512 for model input.
- Generates descriptive prompts for segmentation tasks.
- Provides a PyTorch `SegmentationDataset` class for easy data loading and augmentation.
- Trains a CLIPSeg-based segmentation model for drywall taping and cracks detection.
- Predicts segmentation masks on new images or folders.
- Evaluates predicted masks against ground truth using Dice and IoU metrics.
- Generates visual sanity check overlays of predictions.

---

## Technology Stack

- Python 3.x
- PyTorch for deep learning model training and inference
- Transformers library (HuggingFace) for CLIPSeg model
- OpenCV and PIL for image processing
- Albumentations for data augmentation
- COCO format for dataset annotations
- Matplotlib for visualization
- TQDM for progress bars

---

## Architectural Overview

1. **Dataset Preparation**: Raw COCO datasets are converted to binary masks and resized. Prompts are generated and saved alongside images and masks.

2. **Model Training**: Uses CLIPSeg model fine-tuned on drywall join and cracks datasets with prompts guiding segmentation.

3. **Prediction**: Single image or batch folder prediction scripts generate segmentation masks saved with prompt-based filenames.

4. **Evaluation**: Compares predicted masks with ground truth masks using Dice and IoU metrics, saving visual examples.

---

## Setup Instructions

1. Clone the repository.

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

This project uses separate virtual environments and requirements files for the dataset preprocessing and model training components.

### Dataset Preprocessing Setup

1. Navigate to the `dataset_preprocessing` folder:

```bash
cd dataset_preprocessing
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run dataset preparation and stats generation as described below.

### Model Training Setup

1. Navigate to the `model_training` folder:

```bash
cd model_training
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run training, prediction, and evaluation scripts as described below.

### Root Folder

The root folder contains a `requirements.txt` file but it is not used and can be ignored.

---

### Running the Project

1. Download raw datasets from Roboflow in COCO format and place them under `dataset_preprocessing/raw/`.

2. Run dataset preparation:

```bash
python data_prep.py --dataset drywall_join --raw_folder raw --out_folder data --img_size 512
python data_prep.py --dataset cracks --raw_folder raw --out_folder data --img_size 512
```

3. Generate stats and sanity check previews:

```bash
python generate_stats_and_preview.py
```

4. Train the model:

```bash
python train.py
```

5. Predict on new images or folders:

```bash
python predict.py --image_path path/to/image.jpg --output_folder results
python predict_folder.py --input_folder path/to/images --output_folder results
```

6. Evaluate predictions:

```bash
python evaluate.py --datasets_folder data/cracks/test --output_folder results
```

---

## Notes

- Prompts are fixed per dataset for consistent evaluation.
- Masks and images are resized to 512x512.
- Results folder contains visual overlays for sanity checks.
- Random seeds are fixed for reproducibility.
- Evaluation script expects predicted masks to be pre-generated.

---

## Future Improvements

- Add a requirements.txt file for easier setup.
- Include architectural diagrams for model and data flow.
- Add more detailed usage examples and troubleshooting tips.
- Implement automated testing scripts.

---

## Contact

- Email : shubhramathur1318@gmail.com



