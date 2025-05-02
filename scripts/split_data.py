import os
import shutil
import random

# paths for input and output directories
input_dir = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/processed"
output_dir = "/Users/aoza/pytorch-CycleGAN-and-pix2pix/datasets/my_dataset/"

# create subdirectories for train and val sets
os.makedirs(os.path.join(output_dir, "train/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train/ct"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/ct"), exist_ok=True)

# retrieve patient IDs from preprocessed file names
patients = set(f.split("_")[0] for f in os.listdir(input_dir) if f.endswith(".nii.gz"))

# shuffle the list and split into training and validation sets (80% train, 20% val)
random.seed(42)  # fix random seed for reproducibility
patients = list(patients)
random.shuffle(patients)
split_idx = int(len(patients) * 0.8)
train_patients = patients[:split_idx]
val_patients = patients[split_idx:]

# function to copy and rename files for the given patient list and folder
def process_files(patient_list, folder_name):
    for patient_id in patient_list:
        # process the MRI file
        mri_file = f"{patient_id}_mr_resampled.nii.gz"
        if os.path.exists(os.path.join(input_dir, mri_file)):
            shutil.copy(
                os.path.join(input_dir, mri_file),
                os.path.join(output_dir, folder_name, "mri", f"mri_{patient_id}.nii.gz")
            )
        #process the CT file
        ct_file = f"{patient_id}_ct_resampled.nii.gz"
        if os.path.exists(os.path.join(input_dir, ct_file)):
            shutil.copy(
                os.path.join(input_dir, ct_file),
                os.path.join(output_dir, folder_name, "ct", f"ct_{patient_id}.nii.gz")
            )

# copy and rename files for both training and validation sets
process_files(train_patients, "train")
process_files(val_patients, "val")

print("Splitting and renaming completed!")