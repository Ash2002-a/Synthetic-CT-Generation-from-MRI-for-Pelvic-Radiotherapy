import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

# paths to the directories for generated and ground truth images (MRI -> Generated, CT -> Ground Truth)
generated_dir = "/Users/aoza/Mphy0043_cw/results"  
ground_truth_dir = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/processed/CT"  

# initialize lists to store metrics
mae_list = []
psnr_list = []
ssim_list = []

# get sorted file lists
generated_files = sorted(os.listdir(generated_dir))
ground_truth_files = sorted(os.listdir(ground_truth_dir))

# if there's a mismatch, we will take the minimum length between the two
num_images = min(len(generated_files), len(ground_truth_files))

# loop through images and compute metrics for the number of valid pairs
for i in range(num_images):
    gen_file = generated_files[i]
    gt_file = ground_truth_files[i]

    # load images
    gen_path = os.path.join(generated_dir, gen_file)
    gt_path = os.path.join(ground_truth_dir, gt_file)

    # ensure the files are image files (e.g., .png or .jpg)
    if gen_file.endswith(('.png', '.jpg', '.jpeg')) and gt_file.endswith(('.png', '.jpg', '.jpeg')):
        gen_img = np.array(Image.open(gen_path).convert("L"), dtype=np.float32)
        gt_img = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)

        # resize the generated image to match the ground truth image size
        gen_img_resized = np.array(Image.fromarray(gen_img).resize(gt_img.shape[::-1]), dtype=np.float32)

        # compute metrics
        mae = np.mean(np.abs(gen_img_resized - gt_img))
        psnr_value = psnr(gt_img, gen_img_resized, data_range=gt_img.max() - gt_img.min())
        ssim_value = ssim(gt_img, gen_img_resized, data_range=gt_img.max() - gt_img.min())

        # append results to the lists
        mae_list.append(mae)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

        # print individual results for each imaage pair
        print(f"Image: {gen_file}")
        print(f"MAE: {mae:.4f}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}\n")

# compute average metrics for all images
mean_mae = np.mean(mae_list)
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)

# Print average results
print("Average Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mean_mae:.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {mean_psnr:.4f}")
print(f"Structural Similarity Index Measure (SSIM): {mean_ssim:.4f}")
