import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from typing import List

SAMPLING_FRAC = 0.0179

AUG_ROOT_DIR = '/storage/project/r-smussmann3-0/aqin32/chexpert/train_augmented_n{N_AUGMENTATIONS}'

if not os.path.isdir(AUG_ROOT_DIR):
    print(f"CRITICAL WARNING: Augmented data directory not found at {AUG_ROOT_DIR}. Did you run the generation script?")

# --- 1. Augmented Paths DataFrame Generation and Sampling ---

def generate_augmented_paths_df(aug_root_dir: str) -> pd.DataFrame:
    """
    Scans the augmented image directory structure and generates a DataFrame
    listing all augmented image paths and their corresponding original sequence ID.
    """
    if not os.path.isdir(aug_root_dir):
        return pd.DataFrame()

    data = []
    
    # original_ids are the patient_study subdirectories (e.g., 'patient00001_study1')
    original_ids = [d for d in os.listdir(aug_root_dir) if os.path.isdir(os.path.join(aug_root_dir, d))]

    # This loop is slow for very large datasets, but comprehensive for initial setup
    for original_id in tqdm(original_ids, desc="Scanning Augmented Data"):
        id_dir = os.path.join(aug_root_dir, original_id)
        
        aug_files = [f for f in os.listdir(id_dir) if f.endswith('.jpg')]
        
        for filename in aug_files:
            full_path = os.path.join(id_dir, filename)
            data.append([original_id, full_path])

    df_aug = pd.DataFrame(data, columns=['Original_ID', 'Aug_Path'])
    return df_aug

# Generate the full DataFrame of augmented paths
df_augmented_paths = generate_augmented_paths_df(AUG_ROOT_DIR)

if df_augmented_paths.empty:
    raise RuntimeError("Could not generate augmented paths DataFrame. Check AUG_ROOT_DIR.")

print(f"\nFull Augmented Paths DataFrame generated with {len(df_augmented_paths)} entries.")

# --- Sampling Logic ---
# Find all unique Original IDs
unique_ids = df_augmented_paths['Original_ID'].unique()
df_unique_ids = pd.DataFrame(unique_ids, columns=['Original_ID'])

# Sample a fraction of these unique IDs
sampled_unique_ids_df = df_unique_ids.sample(frac=SAMPLING_FRAC, random_state=42).reset_index(drop=True)
sampled_ids_list = sampled_unique_ids_df['Original_ID'].tolist()

print(f"Sampling {SAMPLING_FRAC*100:.1f}% of unique sequences.")
print(f"Total unique sequences found: {len(unique_ids)}")
print(f"Sequences to be used: {len(sampled_ids_list)}")


# --- 2. Runtime Transform ---
runtime_transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# --- 3. Custom Dataset Class (Modified to accept pre-sampled IDs) ---
class SequenceDataset(Dataset):
    def __init__(self, aug_root_dir: str, original_ids_list: List[str], transform):
        """
        Loads sequences of augmented images from disk using a pre-filtered list of IDs.
        """
        self.aug_root_dir = aug_root_dir
        self.original_ids = original_ids_list
        self.transform = transform
        self.expected_n_aug = 10

    def __len__(self):
        return len(self.original_ids)

    def __getitem__(self, idx):
        # 1. Get the pre-sampled original image identifier 
        original_id = self.original_ids[idx]
        id_dir = os.path.join(self.aug_root_dir, original_id)
        
        # 2. Load all augmented image paths for this original image
        aug_files = sorted([os.path.join(id_dir, f) for f in os.listdir(id_dir) if f.endswith('.jpg')])
        
        # --- File Count Check (Still useful if the generation was interrupted) ---
        if len(aug_files) != self.expected_n_aug:
             print(f"🚨 DEBUG WARNING: ID {original_id} expected {self.expected_n_aug} files but found {len(aug_files)}. Loading found files.")
        
        sequence_tensors = []
        for file_path in aug_files:
            try:
                img = Image.open(file_path).convert('RGB')
                tensor = self.transform(img)
                sequence_tensors.append(tensor)
            except Exception as e:
                # Log the error but continue if other files exist
                print(f"❌ DEBUG ERROR: Failed to load/transform file {file_path} for ID {original_id}. Error: {e}")
                continue
            
        if not sequence_tensors:
             raise RuntimeError(f"CRITICAL ERROR: No valid images loaded for ID: {original_id}. Sequence skipped.")
             
        sequence_tensor = torch.stack(sequence_tensors)
        
        return sequence_tensor, original_id

# --- 4. Instantiate the Dataset and DataLoader ---

sequence_dataset = SequenceDataset(
    AUG_ROOT_DIR,
    original_ids_list=sampled_ids_list, # Use the sampled list
    transform=runtime_transform
)

# DataLoader parameters
SEQ_BATCH_SIZE = 1
SEQ_NUM_WORKERS = 4 

seq_train_loader = DataLoader(
    sequence_dataset, 
    batch_size=SEQ_BATCH_SIZE, 
    shuffle=True, 
    num_workers=SEQ_NUM_WORKERS, 
    pin_memory=True
)

print(f"\nAugmented Sequence DataLoader ready: {len(sequence_dataset)} sequences loaded for training.")
print(f"Sequence DataLoader batch size: {SEQ_BATCH_SIZE}")


# --- 5. Running DataLoader Integrity Check ---

print("\n--- Running DataLoader Integrity Check (First 5 Batches) ---")
try:
    for sequence, original_id in tqdm(seq_traiwloader):
        # sequence.shape: (1, N_loaded, C, H, W)
        
        # Remove the batch dimension
        sequence_squeezed = sequence.squeeze(0) # shape (N_loaded, C, H, W)

        # Final check on loaded sequence count
        if sequence_squeezed.shape[0] != 10:
             print(f"🚨 Final Check Warning: ID {original_id[0]} loaded {sequence_squeezed.shape[0]} images, expected 10.")

        print(sequence_squeezed.shape)

            
except RuntimeError as e:
    print(f"\n❌ CRITICAL DATALOADER STOP: {e}")
except Exception as e:
    print(f"\n❌ UNHANDLED DATALOADER ERROR: {e}")

print("--- Integrity Check Complete ---")