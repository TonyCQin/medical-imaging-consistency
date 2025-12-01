import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as v2
import torch

# --- Configuration ---
dir = '/storage/home/hcoda1/2/aqin32/scratch/chexpert'
CSV_TRAIN_PATH = os.path.join(dir, "train.csv")
CSV_VALID_PATH = os.path.join(dir, "valid.csv")

REDUNDANT_PREFIX = "CheXpert-v1.0-small/"
print(dir)
current_directory = os.getcwd()
print(current_directory)

def load_and_correct_paths(csv_path):
    print(csv_path)
    """Loads a CSV, corrects image paths to be absolute on the Colab disk, and returns the DataFrame."""

    df = pd.read_csv(csv_path)
    df['Path'] = df['Path'].str.replace(REDUNDANT_PREFIX, '', regex=False)
    df['Path'] = dir + '/' + df['Path'] 
    return df

# Load and correct the train and validation DataFrames
train_df = load_and_correct_paths(CSV_TRAIN_PATH)
valid_df = load_and_correct_paths(CSV_VALID_PATH)

print(f"Train DataFrame loaded: {len(train_df)} rows")
print(f"Valid DataFrame loaded: {len(valid_df)} rows")

# Set the desired number of augmented images per original image
N_AUGMENTATIONS = 10

# ex csv entry
# CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,Female,68,Frontal,AP,1.0,,,,,,,,,0.0,,,,1.0

# Define the root directory for all augmented images
aug_dir = '/storage/project/r-smussmann3-0/aqin32/chexpert'
AUG_ROOT_DIR = os.path.join(aug_dir, 'train_augmented_n{N_AUGMENTATIONS}')
os.makedirs(AUG_ROOT_DIR, exist_ok=True)
print(f"Augmented images will be saved in: {AUG_ROOT_DIR}")

# --- Augmentation Pipeline ---
# Define a strong augmentation pipeline using torchvision.transforms.v2
# This pipeline is composed of several common image augmentations for robust training.
# v2 transformations handle PIL Images and Tensors seamlessly.
strong_augmentations = v2.Compose([
    v2.RandomResizedCrop(size=224, scale=(0.8, 1.0)), # Randomly crop and resize
    v2.RandomHorizontalFlip(p=0.5), # Flip 50% of the time
    v2.RandomRotation(degrees=10), # Rotate by up to +/- 10 degrees
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Jitter colors
    v2.ToTensor(), # Convert to Tensor
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (using ImageNet stats)
])


def generate_augmentations(image_path: str, output_root: str, n: int, transform: v2.Compose):
    """
    Loads an image, generates N augmented versions, and saves them
    into a numerically named subdirectory.
    
    Args:
        image_path: Absolute path to the original image.
        output_root: The base directory where augmented images will be saved.
        n: The number of augmented images to generate.
        transform: The torchvision transformation pipeline.
    """
    # 1. Create a unique, numerical subdirectory name
    # We'll use the DataFrame index (which is an integer) to name the subfolder.
    # Note: If your DataFrame index isn't unique or sequential, you might need 
    # to create a new unique ID column (e.g., df['ID'] = range(len(df))).
    dirname = os.path.dirname(image_path)
    view = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    study_id = os.path.basename(dirname)
    patient_dir = os.path.basename(os.path.dirname(dirname))
    # print(view)
    # print(study_id)
    # print(patient_dir)

    unique_id = f"{patient_dir}_{study_id}_{view}"
    
    # Use a cleaner directory structure: base_dir / unique_id / aug_1.jpg
    target_dir = os.path.join(output_root, unique_id)
    os.makedirs(target_dir, exist_ok=True)

    try:
        # 2. Load the original image
        img = Image.open(image_path).convert('RGB')
        
        # 3. Generate and save N augmentations
        for i in range(n):
            # --- Recommended for File Saving (Not Training Tensors) ---
            file_save_transform = v2.Compose([
                v2.RandomResizedCrop(size=224, scale=(0.8, 1.0)), 
                v2.RandomHorizontalFlip(p=0.5), 
                v2.RandomRotation(degrees=10), 
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
            
            aug_img_pil = file_save_transform(img)
            
            # Save the augmented image
            save_path = os.path.join(target_dir, f'aug_{i}.jpg')
            aug_img_pil.save(save_path)
            
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}. Skipping.")
    except Exception as e:
        print(f"An error occurred processing {image_path}: {e}. Skipping.")


# --- Main Processing Loop ---
print(f"Starting augmentation generation for {len(train_df)} images (N={N_AUGMENTATIONS} augmentations per image)...")

# Use tqdm to show a progress bar
# Setting disable=False to ensure the progress bar is shown
tqdm.pandas(desc="Generating Augmentations") 

# Apply the function to every row in the training DataFrame
# The `apply` method is used here for simplicity; for very large datasets, 
# you might use multiprocessing or Dask for better performance.
train_df.progress_apply(
    lambda row: generate_augmentations(
        image_path=row['Path'], 
        output_root=AUG_ROOT_DIR, 
        n=N_AUGMENTATIONS, 
        transform=strong_augmentations # Note: transform is not used internally, see function comment
    ),
    axis=1
)

print("Augmentation generation complete! \n")
print(f"Check {AUG_ROOT_DIR} for the augmented images.")