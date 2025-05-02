import os
import shutil

# define the directories
source_dir = "/Users/aoza/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images"
generated_dir = "/Users/aoza/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/fake_B"
ground_truth_dir = "/Users/aoza/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/real_B"

# create target directories if they don't exist
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(ground_truth_dir, exist_ok=True)

# move files to the respective directories
for filename in os.listdir(source_dir):
    # skip directories
    if os.path.isdir(os.path.join(source_dir, filename)):
        continue
    if "fake_B" in filename:
        shutil.move(os.path.join(source_dir, filename), os.path.join(generated_dir, filename))
    elif "real_B" in filename:
        shutil.move(os.path.join(source_dir, filename), os.path.join(ground_truth_dir, filename))

print("Files have been sorted.")