import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from PIL import Image

# set up directories
BASE_DIR = '/Users/aoza/mphy0043_cw'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')  # directory containing the Pix2Pix output PNGs
EVAL_DIR = os.path.join(BASE_DIR, 'pix2pix_evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

class Pix2PixEvaluator:
    def __init__(self, results_dir, eval_dir):
        self.results_dir = results_dir
        self.eval_dir = os.path.join(eval_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(os.path.join(self.eval_dir, 'comparisons'), exist_ok=True)

    @staticmethod
    def normalize(image):
        """Normalize image to [0, 1] range"""
        if isinstance(image, np.ndarray):
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            return (image - p1) / (p99 - p1)
        return image

    @staticmethod
    def compute_metrics(pred, target):
        """Compute the same metrics as in the autoencoder evaluation"""
        noise_std = np.std(target - pred)
        
        return {
            'psnr': psnr(target, pred, data_range=1.0),
            'ssim': ssim(target, pred, data_range=1.0),
            'mae': np.mean(np.abs(pred - target)),
            'mse': np.mean((pred - target) ** 2),
            'nrmse': np.sqrt(np.mean((pred - target) ** 2)) / (target.max() - target.min()),
            'ncc': pearsonr(pred.flatten(), target.flatten())[0],
            'gradient_error': np.mean(np.abs(gaussian_gradient_magnitude(pred, 1.0) - gaussian_gradient_magnitude(target, 1.0))),
            'snr': 20 * np.log10(np.mean(np.abs(target)) / (noise_std + 1e-8)),
            'cnr': np.abs(np.percentile(target, 90) - np.percentile(target, 10)) / (noise_std + 1e-8)
        }

    def process_image(self, image_path, epoch):
        """Process a single comparison image containing MRI, generated CT, and real CT"""
        # load and split the comparison image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # convert to grayscale if necessary
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # the image is stacked vertically, so split it into three equal parts
        height = img_array.shape[0]
        split_height = height // 3
        
        mri = img_array[:split_height]
        generated_ct = img_array[split_height:2*split_height]
        real_ct = img_array[2*split_height:]
        
        # normalize the images
        generated_ct = self.normalize(generated_ct)
        real_ct = self.normalize(real_ct)
        
        # compute metrics
        metrics = self.compute_metrics(generated_ct, real_ct)
        metrics['epoch'] = epoch
        
        # save the comparison
        self.save_comparison(mri, generated_ct, real_ct, epoch)
        
        return metrics

    def save_comparison(self, mri, generated_ct, real_ct, epoch):
        """Save the comparison visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(axes, [mri, generated_ct, real_ct], 
                                ['MRI Input', 'Generated CT', 'Real CT']):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.savefig(os.path.join(self.eval_dir, 'comparisons', f'comparison_epoch_{epoch}.png'))
        plt.close()

    def evaluate_results(self):
        """Evaluate all result images"""
        all_metrics = []
        image_files = sorted([f for f in os.listdir(self.results_dir) if f.startswith('epoch_') and f.endswith('.png')])
        
        for img_file in tqdm(image_files, desc="Processing Results"):
            epoch = int(img_file.split('_')[1].split('.')[0])
            metrics = self.process_image(os.path.join(self.results_dir, img_file), epoch)
            all_metrics.append(metrics)
        
        # convert to DataFrame and save results
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(self.eval_dir, 'metrics.csv'), index=False)
        self.plot_metrics(df)
        self.save_summary(df)
        
        return df

    def plot_metrics(self, df):
        """Plot metrics over epochs"""
        metrics = ['psnr', 'ssim', 'mae', 'mse', 'nrmse', 'ncc', 'gradient_error', 'snr', 'cnr']
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        
        for ax, metric in zip(axes.flatten(), metrics):
            sns.lineplot(data=df, x='epoch', y=metric, ax=ax)
            ax.set_title(f'{metric.upper()} vs Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, 'metrics_over_epochs.png'))
        plt.close()

    def save_summary(self, df):
        """Save statistical summary of metrics"""
        metrics = ['psnr', 'ssim', 'mae', 'mse', 'nrmse', 'ncc', 'gradient_error', 'snr', 'cnr']
        summary = {metric: df[metric].describe().to_dict() for metric in metrics}
        
        with open(os.path.join(self.eval_dir, 'summary.txt'), 'w') as f:
            f.write("Pix2Pix Evaluation Summary\n")
            f.write("=" * 30 + "\n\n")
            
            for metric, stats in summary.items():
                f.write(f"{metric.upper()} Statistics:\n")
                f.write("-" * 20 + "\n")
                for stat, val in stats.items():
                    f.write(f"{stat}: {val:.4f}\n")
                f.write("\n")

def main():
    try:
        print("Initializing Pix2Pix evaluator...")
        evaluator = Pix2PixEvaluator(RESULTS_DIR, EVAL_DIR)
        
        print("Starting evaluation...")
        df = evaluator.evaluate_results()
        
        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {EVAL_DIR}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()