import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

# Paths to the directories
generated_dir = "/Users/aoza/mphy0043_cw/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/fake_B"
ground_truth_dir = "/Users/aoza/mphy0043_cw/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/real_B"

# initialize metrics
mae_list = []
psnr_list = []
ssim_list = []

# get sorted file lists
generated_files = sorted(os.listdir(generated_dir))
ground_truth_files = sorted(os.listdir(ground_truth_dir))

# ensure the directories have the same number of files
assert len(generated_files) == len(ground_truth_files), "Mismatch in number of images!"

# loop through images and compute metrics
for gen_file, gt_file in zip(generated_files, ground_truth_files):
    # load images
    gen_path = os.path.join(generated_dir, gen_file)
    gt_path = os.path.join(ground_truth_dir, gt_file)

    gen_img = np.array(Image.open(gen_path).convert("L"), dtype=np.float32)
    gt_img = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)

    # compute metrics
    mae = np.mean(np.abs(gen_img - gt_img))
    psnr_value = psnr(gt_img, gen_img, data_range=gt_img.max() - gt_img.min())
    ssim_value = ssim(gt_img, gen_img, data_range=gt_img.max() - gt_img.min())

    # append results
    mae_list.append(mae)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)

    # print individual results for each image pair
    print(f"Image: {gen_file}")
    print(f"MAE: {mae:.4f}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}\n")

# Compute average metrics
mean_mae = np.mean(mae_list)
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)

# print average results
print("Average Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mean_mae:.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {mean_psnr:.4f}")
print(f"Structural Similarity Index Measure (SSIM): {mean_ssim:.4f}")