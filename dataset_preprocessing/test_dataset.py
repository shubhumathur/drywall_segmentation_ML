from dataset_preprocessing.data_prep import SegmentationDataset
import albumentations as A

# optional transforms
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], additional_targets={'mask':'mask'})

# create dataset instance
ds = SegmentationDataset(base_folder='dataset_preprocessing/data/drywall_join', split='train', transform=transform)

# check length
print("Number of samples:", len(ds))

# get one sample
img, mask, prompt = ds[0]
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Prompt string:", prompt)
