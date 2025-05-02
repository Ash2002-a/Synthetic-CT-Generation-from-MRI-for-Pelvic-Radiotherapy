import os
import nibabel as nib
import numpy as np
from PIL import Image

# paths
input_dir = "/Users/aoza/pytorch-CycleGAN-and-pix2pix/datasets/my_dataset/"
output_dir = "/Users/aoza/pytorch-CycleGAN-and-pix2pix/datasets/my_dataset_png/"

# create output directories
os.makedirs(os.path.join(output_dir, "train/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train/ct"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/ct"), exist_ok=True)

# function to normalise and convert to PNG
def nii_to_png(input_path, output_path):
    # load NIfTI file
    nii = nib.load(input_path)
    data = nii.get_fdata()

    # normalize to [0, 255]
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
    data = data.astype(np.uint8)

    # save each slice as a PNG
    for i in range(data.shape[2]):  # Loop over slices
        img = Image.fromarray(data[:, :, i])
        img.save(os.path.join(output_path, f"{os.path.basename(output_path)}_slice{i:03d}.png"))

# process train and val folders
for split in ["train", "val"]:
    for modality in ["mri", "ct"]:
        input_path = os.path.join(input_dir, split, modality)
        output_path = os.path.join(output_dir, split, modality)
        for file in os.listdir(input_path):
            if file.endswith(".nii.gz"):
                nii_to_png(os.path.join(input_path, file), output_path)

print("Conversion to PNG completed!")