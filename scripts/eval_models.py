import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(gt_image, generated_image):
    """Compute MAE, PSNR, and SSIM between ground truth and generated images."""
     # convert images to numpy arrays and normalize pixel values to [0,1]
    gt_np = np.array(gt_image, dtype=np.float32) / 255.0
    gen_np = np.array(generated_image, dtype=np.float32) / 255.0
    
    # compute mean absolute error (mae)
    mae = np.mean(np.abs(gt_np - gen_np))
    
    # cmpute peak signal-to-noise ratio (psnr)
    psnr_value = psnr(gt_np, gen_np, data_range=1.0)
    
    # compute structural similarity index (ssim)
    ssim_value = ssim(gt_np, gen_np, data_range=1.0, multichannel=True)
    
    return mae, psnr_value, ssim_value


def evaluate_model(results_dir, gt_dir, label):
    """
    Evaluate a model by comparing generated images to ground truth images.
    Updates the metrics dynamically.
    """
    # initialize lists to store metric values
    mae_scores, psnr_scores, ssim_scores = [], [], []
    
     # get list of image filenames in the results directory
    filenames = sorted(os.listdir(results_dir))
    total_images = len(filenames)

    # check if there are images to process
    if total_images == 0:
        print(f"[{label}] No images found in {results_dir}. Skipping...")
        return [], [], []

    start_time = time.time()
     # iterate through each image file
    for idx, filename in enumerate(filenames):

        # construct full paths for ground truth and generated images
        gt_path = os.path.join(gt_dir, filename)
        gen_path = os.path.join(results_dir, filename)

        # check if the ground truth image exists
        if os.path.exists(gt_path):
            gt_image = Image.open(gt_path).convert('RGB')
            gen_image = Image.open(gen_path).convert('RGB')
            
            #  resize generated image to match ground truth dimensions
            gen_image = gen_image.resize(gt_image.size, Image.BILINEAR)
            
            # Compute metrics
            mae, psnr_value, ssim_value = calculate_metrics(gt_image, gen_image)
            mae_scores.append(mae)
            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)

        # Print progress every 10 images
        if (idx + 1) % 10 == 0 or idx == total_images - 1:
            elapsed = time.time() - start_time
            avg_time_per_image = elapsed / (idx + 1)
            remaining_time = avg_time_per_image * (total_images - idx - 1)
            print(f"[{label}] Processed {idx + 1}/{total_images} images... "
                  f"Estimated time left: {remaining_time:.2f} seconds")

    # calculate total execution time
    total_time = time.time() - start_time
    print(f"[{label}] Evaluation complete! Total time: {total_time:.2f} seconds")
    print(f"MAE: {mae:.4f}, PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    
    return mae_scores, psnr_scores, ssim_scores


def plot_metrics(metrics_pix2pix, metrics_autoencoder):
    """Plot comparison graphs for MAE, PSNR, and SSIM after both evaluations."""
    epochs = range(1, len(metrics_pix2pix[0]) + 1)
    epochs_auto = range(1, len(metrics_autoencoder[0]) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # MAE plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics_pix2pix[0], label='Pix2Pix', marker='o')
    plt.plot(epochs_auto, metrics_autoencoder[0], label='Autoencoder', marker='s')
    plt.xlabel('Image Index')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    
    # PSNR plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics_pix2pix[1], label='Pix2Pix', marker='o')
    plt.plot(epochs_auto, metrics_autoencoder[1], label='Autoencoder', marker='s')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (dB)')
    plt.title('Peak Signal-to-Noise Ratio')
    plt.legend()
    
    # SSIM plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics_pix2pix[2], label='Pix2Pix', marker='o')
    plt.plot(epochs_auto, metrics_autoencoder[2], label='Autoencoder', marker='s')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity Index')
    plt.legend()
    
    plt.tight_layout()
    plt.show()  # show after both evaluations


def main():
    PIX2PIX_RESULTS = '/Users/aoza/Mphy0043_cw/results'  
    AUTOENCODER_RESULTS = '/Users/aoza/Mphy0043_cw/autoencoder_output' 
    GROUND_TRUTH_DIR = '/Users/aoza/mphy0043_cw/Data/mphy0043_cw/post/Task1/Pelvis' 

    print("Evaluating Pix2Pix...")
    metrics_pix2pix = evaluate_model(PIX2PIX_RESULTS, GROUND_TRUTH_DIR, "Pix2Pix")

    print("Evaluating Autoencoder...")
    metrics_autoencoder = evaluate_model(AUTOENCODER_RESULTS, GROUND_TRUTH_DIR, "Autoencoder")

    print("Plotting comparison results...")
    plot_metrics(metrics_pix2pix, metrics_autoencoder)


if __name__ == '__main__':
    main()
