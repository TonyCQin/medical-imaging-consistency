import os
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pandas import json_normalize
import json
import argparse
from itertools import cycle
import cv2
from torchvision.transforms import v2
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm.auto import tqdm
import torchvision.models as models
import timm
from PIL import Image


# parser = argparse.ArgumentParser(description='Train ResNet on Camera Traps dataset with consistency loss.')
# parser.add_argument('--alpha', type=float, default=1.0, help='Weight for supervised loss')
# parser.add_argument('--beta', type=float, default=0.1, help='Weight for consistency loss')
# parser.add_argument('--start', type=int, default=0, help='Epoch to start considering consistency loss')
# parser.add_argument('--confidence', type=float, default=0, help='Epoch to start considering consistency loss')
# args = parser.parse_args()

# alpha = args.alpha
# beta = args.beta
# start = args.start
# confidence = args.confidence
# # ### Setting up image directories and splits
# print(alpha, beta, start, confidence)
# %%
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

print("\n--- 5. Final Image Verification Test ---")

test_image_path = valid_df['Path'].iloc[0]

# cnn transforms, prefer different transforms than vit transforms
cnn_transforms = transforms.Compose([
    transforms.Resize(256),         # Resize shortest side to 256
    transforms.CenterCrop(224),     # Crop to standard 224x224 input size
    transforms.ToTensor(),          # Convert image to a tensor (H, W, C -> C, H, W)
    transforms.Normalize(           # Standard ImageNet normalization (a good starting point)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

vit_transforms = transforms.Compose([
    transforms.Resize(256),         # Resize shortest side to 256
    transforms.CenterCrop(224),     # Crop to standard 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CheXpertDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

        # baseline models benchmark against these five labels normally
        self.labels = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

        # handle uncertain labels
        self.df[self.labels] = self.df[self.labels].replace(-1, 0)
        self.df[self.labels] = self.df[self.labels].fillna(0) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get image
        img_path = self.df.iloc[idx]['Path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # convert to rgb
        image = np.stack([image] * 3, axis=-1)

        
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # 3. Load Labels
        label_vector = self.df.iloc[idx][self.labels].values.astype(np.float32)
        labels = torch.from_numpy(label_vector)

        return image, labels


# 0.018: 80%, 4,000 sequences, 40,000 images
# 0.02 90%, 4,500 sequences, 45,000 images
# 0.0157: 70%, 3,500 sequences, 35,000 images  
# 0.0112: 50%, 2,500 sequence, 25,000 images
SAMPLING_FRAC = 0.018

AUG_ROOT_DIR = '/storage/project/r-smussmann3-0/aqin32/chexpert/train_augmented_n{N_AUGMENTATIONS}'

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

# get sequencei ds
unique_ids = df_augmented_paths['Original_ID'].unique()
df_unique_ids = pd.DataFrame(unique_ids, columns=['Original_ID'])
sampled_unique_ids_df = df_unique_ids.sample(frac=SAMPLING_FRAC, random_state=42).reset_index(drop=True)
sampled_ids_list = sampled_unique_ids_df['Original_ID'].tolist()

print(f"Sampling {SAMPLING_FRAC*100:.1f}% of unique sequences.")
print(f"Total unique sequences found: {len(unique_ids)}")
print(f"Sequences to be used: {len(sampled_ids_list)}")


runtime_transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

class SequenceDataset(Dataset):
    def __init__(self, aug_root_dir: str, original_ids_list: list[str], transform):
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
        original_id = self.original_ids[idx]
        id_dir = os.path.join(self.aug_root_dir, original_id)
    
        aug_files = sorted([os.path.join(id_dir, f) for f in os.listdir(id_dir) if f.endswith('.jpg')])
        
        sequence_tensors = []
        for file_path in aug_files:
            img = Image.open(file_path).convert('RGB')
            tensor = self.transform(img)
            sequence_tensors.append(tensor)
            
        if not sequence_tensors:
             raise RuntimeError(f"CRITICAL ERROR: No valid images loaded for ID: {original_id}. Sequence skipped.")
             
        sequence_tensor = torch.stack(sequence_tensors)
        
        return sequence_tensor, original_id


sequence_dataset = SequenceDataset(
    AUG_ROOT_DIR,
    original_ids_list=sampled_ids_list, # Use the sampled list
    transform=runtime_transform
)

# DataLoader parameters
SEQ_BATCH_SIZE = 4
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
    
BATCH_SIZE = 64
NUM_WORKERS = 4

print("Setting up data for CNN model...")
# 0.022: 5000 images, 10%
# 0.067: 15000 images, 30%
# 0.112: 25000 images, 50%
# sample 4.4% of training data to simulate relatively sparse environment (10000 images, 20% of data)

train_df = train_df.sample(frac=0.044, random_state=42).reset_index(drop=True)
cnn_train_dataset = CheXpertDataset(train_df, transform=cnn_transforms)
cnn_valid_dataset = CheXpertDataset(valid_df, transform=cnn_transforms)

cnn_train_loader = DataLoader(
    cnn_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # Speeds up transfer to GPU
)
cnn_valid_loader = DataLoader(
    cnn_valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

vit_train_dataset = CheXpertDataset(train_df, transform=vit_transforms)
vit_valid_dataset = CheXpertDataset(valid_df, transform=vit_transforms)

vit_train_loader = DataLoader(
    vit_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
vit_valid_loader = DataLoader(
    vit_valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

num_classes = 5

## ResNet-50 (Standard PyTorch)
model_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Replace the final fully connected layer (fc)
in_features_resnet = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features_resnet, num_classes)
print(f"ResNet-50 loaded with {num_classes} outputs.")

## EfficientNet-B0 (using timm)
# Use 'tf_efficientnet_b0' for common variant
model_efficientnet = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=num_classes)
print(f"EfficientNet-B0 loaded with {num_classes} outputs.")

## ConvNeXt-Tiny (using timm)
model_convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
print(f"ConvNeXt-Tiny loaded with {num_classes} outputs.")

# ViT (Vision Transformer) - 'vit_base_patch16_224' is a standard starting point
model_vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
print(f"ViT-Base loaded with {num_classes} outputs.")

# swim transformer
model_swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
print(f"Swin-Tiny loaded with {num_classes} outputs.")

# DeiT (Data-efficient Image Transformer) - Uses distillation for better performance
model_deit = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=num_classes)
print(f"DeiT-Base loaded with {num_classes} outputs.")

# CoAtNet (Hybrid CNN-Transformer)
model_coatnet = timm.create_model('coatnet_0_rw_224', pretrained=True, num_classes=num_classes)
print(f"CoAtNet-0 loaded with {num_classes} outputs.")

def high_confidence_consensus_vector(seq_outputs: torch.Tensor, confidence_threshold: float = 0.9) -> torch.Tensor:
    """
    Generates a multi-label pseudo-target vector [5] only if all augmented 
    frames (T) highly agree on the state of each individual label (5 classes).

    Args:
        seq_outputs (torch.Tensor): [T, num_classes] raw model logits.
        confidence_threshold (float): Minimum probability required for the 
                                      consensus prediction on a label.

    Returns:
        torch.Tensor: [num_classes] pseudo-label vector (0.0 or 1.0) if consensus, 
                      or a vector of -1.0 if no consensus for any label.
    """
    num_classes = seq_outputs.size(1)
    T = seq_outputs.size(0)
    
    # [T, num_classes] probabilities
    probabilities = torch.sigmoid(seq_outputs) 

    # [T, num_classes] binary predictions (0 or 1)
    predictions = (probabilities > 0.5).int()
    
    # Initialize the pseudo-target vector
    pseudo_target = torch.full((num_classes,), -1.0, device=seq_outputs.device)

    for i in range(num_classes):
        # check if frames agree
        predictions_i = predictions[:, i] # [T]
        
        # agree if min equiv to max
        perfect_agreement = (predictions_i.min() == predictions_i.max())

        if perfect_agreement:
            consensus_state = predictions_i[0].float() 
            if consensus_state == 1.0:
                min_confidence = probabilities[:, i].min()
                if min_confidence >= confidence_threshold:
                    pseudo_target[i] = 1.0
            
            else:
                max_uncertainty = probabilities[:, i].max()
                if (1.0 - max_uncertainty) >= confidence_threshold:
                    pseudo_target[i] = 0.0

    if torch.all(pseudo_target == -1.0):
        return torch.full((num_classes,), -1.0, device=seq_outputs.device)
        
    return pseudo_target

def majority_vote_consensus_vector(seq_outputs: torch.Tensor, majority_threshold: float = 0.7) -> torch.Tensor:
    """
    Generates a multi-label pseudo-target vector [5] based on a majority vote 
    for each label (0 or 1) across all augmented frames (T).

    Args:
        seq_outputs (torch.Tensor): [T, num_classes] raw model logits.
        majority_threshold (float): Minimum fraction of frames (0.5 to 1.0) 
                                    required to vote for a class state (0 or 1).

    Returns:
        torch.Tensor: [num_classes] pseudo-label vector (0.0 or 1.0).
    """
    num_classes = seq_outputs.size(1)
    T = seq_outputs.size(0)
    probabilities = torch.sigmoid(seq_outputs) 
    predictions = (probabilities > 0.5).float() # Use float for easier summing
    
    positive_votes = predictions.sum(dim=0)
    positive_fraction = positive_votes / T 
    pseudo_target = torch.full((num_classes,), -1.0, device=seq_outputs.device)
    is_positive = (positive_fraction >= majority_threshold)
    is_negative = ((1.0 - positive_fraction) >= majority_threshold)

    pseudo_target[is_positive] = 1.0
    pseudo_target[is_negative] = 0.0
    
    return pseudo_target


def run_experiment(model, train_loader, seq_loader, valid_loader, model_name, num_epochs=20, learning_rate=1e-4, checkpoint_dir='checkpoints'):
    """
    Trains and validates a PyTorch model for multi-label classification.

    Args:
        model (nn.Module): The model architecture (e.g., model_resnet, model_vit).
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
        model_name (str): Unique name for the current model (e.g., 'ResNet50', 'Swin_Tiny').
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Initial learning rate for the optimizer.
        checkpoint_dir (str): Persistent path to save best model weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_auc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n--- Starting Training: {model_name} on {device} ---")
    start_time = time.time()
    epoch_bar = tqdm(range(num_epochs), desc=f"Experiment: {model_name}")
    beta = 0.1
    confidence = 0.75
    for epoch in epoch_bar:
        epoch_start_time = time.time()
        model.train()
        supervised_loss_sum = 0.0
        consistency_loss_sum = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        seq_bar = tqdm(seq_loader)

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.step()

            supervised_loss_sum += loss.item() 
            train_bar.set_postfix(s_loss=loss.item())

        if beta > 0:
            for sequence, original_id in seq_bar:
                sequence = sequence.to(device)
                optimizer.zero_grad()
                batch_size, T, C, H, W = sequence.shape
                inputs_flat = sequence.view(batch_size * T, C, H, W)
                outputs_flat = model(inputs_flat)
                seq_outputs = outputs_flat.view(batch_size, T, -1)

                consistency_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                # for given batch stores how many sequences were confident
                consistent_sequences = 0 
                for i in range(batch_size):
                    # pseudo_target_vector = high_confidence_consensus_vector(seq_outputs[i].detach(), confidence)
                    pseudo_target_vector = majority_vote_consensus_vector(seq_outputs[i].detach(), confidence)
                    
                    # if one of the predicted labels is viable
                    if not torch.all(pseudo_target_vector == -1.0): 
                        consistent_sequences += 1
                        mask = (pseudo_target_vector != -1.0)
                    
                        outputs_i_masked = seq_outputs[i][:, mask] # [T, num_consensus_classes]
                        target_i_masked = pseudo_target_vector[mask] # [num_consensus_classes]
                        targets_repeated = target_i_masked.unsqueeze(0).expand_as(outputs_i_masked)
                        loss = F.mse_loss(outputs_i_masked, targets_repeated, reduction='mean')

                        consistency_loss += loss

                if batch_size > 0 and consistent_sequences > 0:
                    consistency_loss /= consistent_sequences  # normalize per sequence
                    
                total_loss = beta * consistency_loss
                if total_loss.item() > 0:
                    total_loss.backward()               # Backpropagation
                    optimizer.step()              # Update weights
                    consistency_loss_sum += total_loss.item()


        train_loss_supervised = supervised_loss_sum / len(train_loader) if len(train_loader) > 0 else 0.0
        train_loss_consistency = consistency_loss_sum / len(seq_loader) if len(seq_loader) > 0 else 0.0
        train_loss = train_loss_supervised + train_loss_consistency

        model.eval()
        running_val_loss = 0.0
        all_targets = []
        all_predictions = []

        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1} Valid", leave=False)

        with torch.no_grad():
            for inputs, labels in valid_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                probabilities = torch.sigmoid(outputs).cpu().numpy()
                all_targets.append(labels.cpu().numpy())
                all_predictions.append(probabilities)

        val_loss = running_val_loss / len(valid_loader.dataset)

        targets_np = np.concatenate(all_targets)
        predictions_np = np.concatenate(all_predictions)

        try:
            val_auc = roc_auc_score(targets_np, predictions_np, average='macro')
        except ValueError:
            val_auc = 0.0

        scheduler.step(val_loss)

        epoch_bar.set_postfix_str(f"Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

        print(f"Epoch {epoch+1:02d}/{num_epochs} ({time.time() - epoch_start_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Mean AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), checkpoint_file)
            print(f"       ✅ Model checkpoint saved! New Best AUC: {best_val_auc:.4f}")

    total_time = (time.time() - start_time) / 60
    print(f"\nTraining for {model_name} complete. Total time: {total_time:.2f} minutes.")
    print(f"Final Best Validation Mean AUC: {best_val_auc:.4f}")

    return best_val_auc

# cnn models

best_auc_resnet = run_experiment(
    model=model_resnet,
    train_loader=cnn_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=cnn_valid_loader,
    model_name='ResNet50',
    num_epochs=30,
    learning_rate=5e-5 # Lower LR often better for fine-tuning
)

best_auc_efficientnet = run_experiment(
    model=model_efficientnet,
    train_loader=cnn_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=cnn_valid_loader,

    model_name='EfficientNet-B0',
    num_epochs=30,
    learning_rate=5e-5 # Lower LR often better for fine-tuning
)

best_auc_convnext = run_experiment(
    model=model_convnext,
    train_loader=cnn_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=cnn_valid_loader,
    model_name='convnext_tiny',
    num_epochs=30,
    learning_rate=5e-5 # Lower LR often better for fine-tuning
)

best_auc_coatnext = run_experiment(
    model=model_coatnet,
    train_loader=cnn_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=cnn_valid_loader,
    model_name='CoAtNet-0',
    num_epochs=30,
    learning_rate=5e-5 # Lower LR often better for fine-tuning
)

# transformer based models
best_auc_vit = run_experiment(
    model=model_vit,
    train_loader=vit_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=vit_valid_loader,
    model_name='ViT_Base_224',
    num_epochs=30,
    learning_rate=2e-5 # Even lower LR for larger models
)

best_auc_swin = run_experiment(
    model=model_swin,
    train_loader=vit_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=vit_valid_loader,
    model_name='swin_tiny_patch4_window7_224',
    num_epochs=30,
    learning_rate=2e-5 # Even lower LR for larger models
)

best_auc_deit = run_experiment(
    model=model_deit,
    train_loader=vit_train_loader,
    seq_loader=seq_train_loader,
    valid_loader=vit_valid_loader,
    model_name='deit_base_patch16_224',
    num_epochs=30,
    learning_rate=2e-5 # Even lower LR for larger models
)