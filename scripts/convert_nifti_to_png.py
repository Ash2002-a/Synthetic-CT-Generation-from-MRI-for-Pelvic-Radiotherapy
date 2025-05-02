import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from PIL import Image

# define paths for input and output directories
input_dir = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/post/Task1/Pelvis/pelvis"
output_mri_dir = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/processed/MRI"
output_ct_dir = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/processed/CT"

# create output directories if they do not exist
os.makedirs(output_mri_dir, exist_ok=True)
os.makedirs(output_ct_dir, exist_ok=True)

def convert_nifti_to_png(input_path, output_path):
    print(f"Processing: {input_path}")

    # load nifti image using nibabel
    nifti_image = nib.load(input_path)
    image_data = nifti_image.get_fdata()
    
    # select the middle slice along the third dimension
    slice_index = image_data.shape[2] // 2 
    slice_2d = image_data[:, :, slice_index]

    # normalize the pixel values to range [0, 255]
    normalized_img = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255

    # convert numpy array to image and save as png
    img = Image.fromarray(normalized_img.astype(np.uint8))
    img.save(output_path)
    print(f"Saved: {output_path}")

# process MRI and CT images for each patient
for patient_folder in os.listdir(input_dir):
    patient_path = os.path.join(input_dir, patient_folder)

    # check if the current item is a directory (patient folder)
    if os.path.isdir(patient_path):
        mri_path = os.path.join(patient_path, 'mr.nii.gz')
        ct_path = os.path.join(patient_path, 'ct.nii.gz')

        # convert and save mri image if it exists
        if os.path.exists(mri_path):
            convert_nifti_to_png(mri_path, os.path.join(output_mri_dir, f"{patient_folder}_mri.png"))

         # convert and save ct image if it exists
        if os.path.exists(ct_path):
            convert_nifti_to_png(ct_path, os.path.join(output_ct_dir, f"{patient_folder}_ct.png"))

print("NIfTI to PNG conversion completed!")