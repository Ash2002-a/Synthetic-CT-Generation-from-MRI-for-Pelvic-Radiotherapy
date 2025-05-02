import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.transforms import CenterCrop

# directory paths
BASE_DIR = '/Users/aoza/mphy0043_cw'
ROOT_DIR = os.path.join(BASE_DIR, 'Data/mphy0043_cw/post/Task1/Pelvis')
RESULTS_DIR = os.path.join(BASE_DIR, 'autoencoder_output')
IMG_SIZE = 228

def display_memory_usage():
    # display current memory usage of the process
    import psutil
    process = psutil.Process()
    memory_in_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory Usage: {memory_in_mb:.2f} MB")

class MRI_CT_Dataset(Dataset):
    def __init__(self, files, slice_indices, root_dir):
        super(MRI_CT_Dataset, self).__init__()
        print("Initializing MRI and CT Dataset...")
        self.files = files
        self.root_dir = root_dir
        self.slice_indices = slice_indices
        self.center_crop = CenterCrop((IMG_SIZE, IMG_SIZE))
        
        # caache to store preprocessed data
        self.data_cache = {}

        print("Preprocessing dataset...")
        for idx, file in enumerate(files):
            if idx % 10 == 0:
                print(f"Processing file {idx}/{len(files)}")
            file_path = os.path.join(self.root_dir, file)
            mri_path = os.path.join(file_path, "mr.nii.gz")
            ct_path = os.path.join(file_path, "ct.nii.gz")
            
            # load MRI and CT data
            mri_img = nib.load(mri_path)
            ct_img = nib.load(ct_path)
            mri_data = mri_img.get_fdata()
            ct_data = ct_img.get_fdata()

            # normalize volumes to ensure consistent intensity range
            mri_normalized = self.normalize_data(mri_data)
            ct_normalized = self.normalize_data(ct_data)

            # store tensors in cache for quick retrieval
            self.data_cache[file] = {
                'mri': torch.FloatTensor(mri_normalized),
                'ct': torch.FloatTensor(ct_normalized),
                'num_slices': mri_data.shape[2]
            }
        print("Dataset preprocessing complete!")

    def normalize_data(self, data):
        # apply percentile-based normalization to reduce intensity outliers
        lower_percentile = np.percentile(data, 1)
        upper_percentile = np.percentile(data, 99)
        data = np.clip(data, lower_percentile, upper_percentile)
        normalized_data = (data - lower_percentile) / (upper_percentile - lower_percentile)
        return normalized_data.astype(np.float32)

    def __len__(self):
        return self.slice_indices[-1]

    def __getitem__(self, index):
        # identify the corresponding file and slice index for the given index
        image_idx = next(idx for idx in range(1, len(self.slice_indices)) if self.slice_indices[idx-1] <= index <= self.slice_indices[idx]) - 1
        index -= self.slice_indices[image_idx]
        file = self.files[image_idx]
        
        # retrieve cached data
        cached_data = self.data_cache[file]
        slice_idx = min(index, cached_data['num_slices'] - 1)
        mri_slice = cached_data['mri'][:, :, slice_idx]
        ct_slice = cached_data['ct'][:, :, slice_idx]

         # apply center cropping
        mri_slice = self.center_crop(mri_slice)
        ct_slice = self.center_crop(ct_slice)

        return {'mri': mri_slice.unsqueeze(0), 'ct': ct_slice.unsqueeze(0)}

class AutoencoderModel(nn.Module):
    def __init__(self):
        super(AutoencoderModel, self).__init__()

        # encoder layers progressively downsamplle input image
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # decoder layers progressively upsample the encoded image
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # forward pass through encoder layers
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)

        # forward pass through decoder layers with skip connections
        dec3_out = self.dec3(enc3_out)
        dec3_out = F.interpolate(dec3_out, size=enc2_out.shape[2:], mode='bilinear', align_corners=True)
        dec3_out = torch.cat([dec3_out, enc2_out], dim=1)

        dec2_out = self.dec2(dec3_out)
        dec2_out = F.interpolate(dec2_out, size=enc1_out.shape[2:], mode='bilinear', align_corners=True)
        dec2_out = torch.cat([dec2_out, enc1_out], dim=1)

        dec1_out = self.dec1(dec2_out)
        dec1_out = F.interpolate(dec1_out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return dec1_out

def train_model(train_data, val_data, epochs=50, batch_size=4, device='cuda'):
    print("\nTraining started!")
    print(f"Configuration - Epochs: {epochs}, Batch size: {batch_size}, Device: {device}")
    display_memory_usage()

    # create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    model = AutoencoderModel().to(device)
    print("Model initialized and moved to device")

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    print("\nStarting training loop...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        display_memory_usage()

        # training phase
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - ETA: {(time.time() - epoch_start) * (len(train_loader) - batch_idx):.2f}s", end='\r')

            mri_images = batch['mri'].to(device)
            ct_images = batch['ct'].to(device)

            optimizer.zero_grad()
            output = model(mri_images)

            loss_mse = mse_loss(output, ct_images)
            loss_l1 = l1_loss(output, ct_images)
            total_loss = loss_mse + 0.5 * loss_l1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

        print(f"\nTraining loss for epoch {epoch+1}: {train_loss / len(train_loader):.4f}")

        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                mri_images = batch['mri'].to(device)
                ct_images = batch['ct'].to(device)
                output = model(mri_images)
                val_loss += mse_loss(output, ct_images).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss for epoch {epoch+1}: {avg_val_loss:.4f}")

        # early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("New best model found, saving...")
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step()

        # save images every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_mri = mri_images[0].unsqueeze(0)
                sample_ct = ct_images[0].unsqueeze(0)
                sample_output = model(sample_mri)
                comparison = torch.cat([sample_mri, sample_output, sample_ct], dim=2)
                save_image(comparison.cpu(), os.path.join(RESULTS_DIR, f'epoch_{epoch+1}.png'), normalize=True, nrow=1, padding=2)

        print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f}s")

    print(f"Training completed in {time.time() - start_time:.2f}s")
    return model

def main():
    print("Starting the main script...")
    display_memory_usage()

    # fetch file names
    print("\nLoading dataset files...")
    files_list = os.listdir(ROOT_DIR)
    files_list = [file for file in files_list if file not in {'.DS_Store', 'overview'}]
    print(f"Found {len(files_list)} files")

    # caalculate slice indices
    slice_indices = [0]
    for file in files_list:
        file_path = os.path.join(ROOT_DIR, file)
        mri_path = os.path.join(file_path, "mr.nii.gz")
        mri_img = nib.load(mri_path)
        slice_indices.append(slice_indices[-1] + mri_img.shape[2])

    print(f"Total number of slices: {slice_indices[-1]}")

    # create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # create dataset
    print("\nCreating dataset...")
    dataset = MRI_CT_Dataset(files_list, slice_indices, root_dir=ROOT_DIR)
    total_size = len(dataset)
    subset_size = total_size // 8
    indices = torch.randperm(total_size).tolist()[:subset_size]
    dataset_subset = torch.utils.data.Subset(dataset, indices)


    # split dataset into train and validation sets
    print("\nSplitting dataset into train and validation sets...")
    train_size = int(0.8 * len(dataset_subset))
    val_size = len(dataset_subset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset_subset, [train_size, val_size])

    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

    # train the autoencoder model
    print("\nTraining the autoencoder model...")
    trained_model = train_model(train_data, val_data, epochs=50, batch_size=4, device=device)

    # save final model
    print("\nSaving the final model...")
    torch.save(trained_model.state_dict(), os.path.join(RESULTS_DIR, 'autoencoder_final.pth'))
    print("Model training completed!")

if __name__ == '__main__':
    main()
