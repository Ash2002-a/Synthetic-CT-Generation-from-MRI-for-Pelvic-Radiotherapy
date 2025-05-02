import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
from torchvision.utils import save_image
import time
from datetime import datetime

# constants
IMG_SIZE = 228
BATCH_SIZE = 4  
NUM_EPOCHS = 200
BASE_DIR = '/Users/aoza/mphy0043_cw'
DATA_DIR = os.path.join(BASE_DIR, 'Data/mphy0043_cw/post/Task1/Pelvis')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ensure necessary directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = []
        
        print("Loading dataset...")
        for patient_folder in sorted(os.listdir(self.data_dir)):
            if not patient_folder.startswith('.'): # ignore hidden files
                patient_path = os.path.join(self.data_dir, patient_folder)
                if os.path.isdir(patient_path): # ensure it's a directory
                    mri_path = os.path.join(patient_path, 'mr.nii.gz')
                    ct_path = os.path.join(patient_path, 'ct.nii.gz')
                    if os.path.exists(mri_path) and os.path.exists(ct_path):
                        try:
                            # load images and get middle slice
                            mri_img = nib.load(mri_path)
                            ct_img = nib.load(ct_path)
                            
                            slice_idx = mri_img.shape[2] // 2
                            
                            # get the middle slice and normalize
                            mri_slice = self._normalize_slice(mri_img.get_fdata()[:, :, slice_idx])
                            ct_slice = self._normalize_slice(ct_img.get_fdata()[:, :, slice_idx])
                            
                            # convert slices to tensors
                            mri_tensor = torch.FloatTensor(mri_slice).unsqueeze(0)
                            ct_tensor = torch.FloatTensor(ct_slice).unsqueeze(0)
                            
                            # resize images to ensure they match the expected input size
                            target_size = (IMG_SIZE, IMG_SIZE)
                            mri_tensor = F.interpolate(mri_tensor.unsqueeze(0), 
                                                     size=target_size, 
                                                     mode='bilinear', 
                                                     align_corners=True).squeeze(0)
                            ct_tensor = F.interpolate(ct_tensor.unsqueeze(0), 
                                                    size=target_size, 
                                                    mode='bilinear', 
                                                    align_corners=True).squeeze(0)
                            
                            # apply transformations if provided
                            if self.transform:
                                mri_tensor = self.transform(mri_tensor)
                                ct_tensor = self.transform(ct_tensor)
                            
                            # store the MRI-CT image pair
                            self.image_pairs.append({
                                'A': mri_tensor,
                                'B': ct_tensor
                            })
                            print(f"Loaded {patient_folder}")
                        except Exception as e:
                            print(f"Error loading {patient_folder}: {str(e)}")
                            continue
        
        print(f"Successfully loaded {len(self.image_pairs)} image pairs")

    def _normalize_slice(self, slice_data):
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max - slice_min == 0:
            return np.zeros_like(slice_data)
        return (slice_data - slice_min) / (slice_max - slice_min)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        return self.image_pairs[idx]

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(Generator, self).__init__()
        
        # encoder layers (downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # decoder layers (upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # decoder with size checks and adjustments
        u1 = self.up1(d6)
        # make sure u1 matches d5 size
        if u1.size() != d5.size():
            u1 = F.interpolate(u1, size=d5.size()[2:], mode='bilinear', align_corners=True)
            
        u2 = self.up2(torch.cat([u1, d5], 1))
        if u2.size() != d4.size():
            u2 = F.interpolate(u2, size=d4.size()[2:], mode='bilinear', align_corners=True)
            
        u3 = self.up3(torch.cat([u2, d4], 1))
        if u3.size() != d3.size():
            u3 = F.interpolate(u3, size=d3.size()[2:], mode='bilinear', align_corners=True)
            
        u4 = self.up4(torch.cat([u3, d3], 1))
        if u4.size() != d2.size():
            u4 = F.interpolate(u4, size=d2.size()[2:], mode='bilinear', align_corners=True)
            
        u5 = self.up5(torch.cat([u4, d2], 1))
        if u5.size() != d1.size():
            u5 = F.interpolate(u5, size=d1.size()[2:], mode='bilinear', align_corners=True)
            
        return self.final(torch.cat([u5, d1], 1))
        
    def _debug_sizes(self, x):
        # Helper method to debug sizes during forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        print(f"Input size: {x.size()}")
        print(f"d1 size: {d1.size()}")
        print(f"d2 size: {d2.size()}")
        print(f"d3 size: {d3.size()}")
        print(f"d4 size: {d4.size()}")
        print(f"d5 size: {d5.size()}")
        print(f"d6 size: {d6.size()}")
class Discriminator(nn.Module):
    def __init__(self, input_channels=2):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

def train_pix2pix(dataloader, num_epochs=200, device='cuda'):
    # initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # loss functions
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    lambda_L1 = 100
    
    # pptimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # get MRI and CT images
            real_A = batch['A'].to(device)  # MRI
            real_B = batch['B'].to(device)  # CT
            
            # ensure inputs have the correct shape
            if real_A.dim() == 3:
                real_A = real_A.unsqueeze(0)
            if real_B.dim() == 3:
                real_B = real_B.unsqueeze(0)
            
            # train Generator
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            
            # ensure all tensors have the same size before concatenation
            if fake_B.size() != real_B.size():
                fake_B = F.interpolate(fake_B, size=real_B.size()[2:], mode='bilinear', align_corners=True)
            
            pred_fake = discriminator(real_A, fake_B)
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterion_L1(fake_B, real_B) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            
            loss_G.backward()
            optimizer_G.step()
            
            # train Discriminator
            optimizer_D.zero_grad()
            
            pred_real = discriminator(real_A, real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
                
        # save sample images and models every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_B = generator(real_A)
                if fake_B.size() != real_B.size():
                    fake_B = F.interpolate(fake_B, size=real_B.size()[2:], mode='bilinear', align_corners=True)
                img_sample = torch.cat((real_A[0], fake_B[0], real_B[0]), -2)
                save_image(img_sample, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}.png'), normalize=True)
            
            # save models
            torch.save(generator.state_dict(), os.path.join(CHECKPOINTS_DIR, f'generator_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINTS_DIR, f'discriminator_{epoch+1}.pth'))

def main():
    try:
        # set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # create dataset
        print("\nInitializing dataset...")
        dataset = ImageDataset(data_dir=DATA_DIR)
        
        # create dataloader
        print(f"\nCreating DataLoader with batch size {BATCH_SIZE}...")
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"\nDataset size: {len(dataset)} images")
        print(f"Number of batches per epoch: {len(dataloader)}")
        
        # ctart training
        print("\nStarting training...")
        train_pix2pix(dataloader, NUM_EPOCHS, device)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()