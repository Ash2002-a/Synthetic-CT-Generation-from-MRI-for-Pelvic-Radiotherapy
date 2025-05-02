import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from autoencoder import AutoencoderModel

# set up directories
BASE_DIR = '/Users/aoza/mphy0043_cw'
ROOT_DIR = os.path.join(BASE_DIR, 'Data/mphy0043_cw/post/Task1/Pelvis')
RESULTS_DIR = os.path.join(BASE_DIR, 'autoencoder_evaluation')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'autoencoder_output')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(device):
    # load the trained autoencoder model from a checkpoint
    model = AutoencoderModel().to(device)
    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, 'autoencoder_final.pth'), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()# set model to evaluation mode
    return model

class AutoencoderEvaluator:
    def __init__(self, model, device, results_dir):
        # initialize evaluator with model, device, and results directory
        self.model = model
        self.device = device
        self.results_dir = os.path.join(results_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def normalize(volume):
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        return (volume - p1) / (p99 - p1)

    @staticmethod
    def compute_metrics(pred, target):
        pred_np, target_np = pred.cpu().numpy().squeeze(), target.cpu().numpy().squeeze()
        noise_std = np.std(target_np - pred_np)
        
        # compute multiple image similarity metrics
        return {
            'psnr': psnr(target_np, pred_np, data_range=1.0),
            'ssim': ssim(target_np, pred_np, data_range=1.0),
            'mae': np.mean(np.abs(pred_np - target_np)),
            'mse': np.mean((pred_np - target_np) ** 2),
            'nrmse': np.sqrt(np.mean((pred_np - target_np) ** 2)) / (target_np.max() - target_np.min()),
            'ncc': pearsonr(pred_np.flatten(), target_np.flatten())[0],
            'gradient_error': np.mean(np.abs(gaussian_gradient_magnitude(pred_np, 1.0) - gaussian_gradient_magnitude(target_np, 1.0))),
            'snr': 20 * np.log10(np.mean(np.abs(target_np)) / (noise_std + 1e-8)),
            'cnr': np.abs(np.percentile(target_np, 90) - np.percentile(target_np, 10)) / (noise_std + 1e-8)
        }

    def evaluate_slice(self, mri_slice, ct_slice, patient_id, slice_idx):
        # forward pass through the model to generate a predicted ct slice
        mri_slice, ct_slice = mri_slice.to(self.device), ct_slice.to(self.device)
        with torch.no_grad():
            pred = self.model(mri_slice)
        
        # save visual comparisons for every 10th slice
        if slice_idx % 10 == 0:
            self.save_comparison(mri_slice, ct_slice, pred, patient_id, slice_idx)
        
        # compute metrics for the current slice
        metrics = self.compute_metrics(pred, ct_slice)
        metrics['slice_idx'] = slice_idx
        return metrics

    def save_comparison(self, mri, ct, pred, patient_id, slice_idx):
        # visualise and save comparisons of mri input, generated ct, and real ct
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(axes, [mri, pred, ct], ['MRI Input', 'Generated CT', 'Real CT']):
            ax.imshow(img.cpu().squeeze(), cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        # create directory for patient comparisons and save the figure
        save_path = os.path.join(self.results_dir, 'comparisons', f'patient_{patient_id}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'slice_{slice_idx}.png'))
        plt.close()

    def process_volume(self, mri_path, ct_path, patient_id):
         # load and normalize mri and ct volumes
        mri_vol, ct_vol = self.normalize(nib.load(mri_path).get_fdata()), self.normalize(nib.load(ct_path).get_fdata())
        metrics_list = []
        
        # iterate through all slices in the volume
        for idx in range(mri_vol.shape[2]):
            mri_slice = torch.FloatTensor(mri_vol[:, :, idx]).unsqueeze(0).unsqueeze(0)
            ct_slice = torch.FloatTensor(ct_vol[:, :, idx]).unsqueeze(0).unsqueeze(0)
            
            # resize slices to match model input dimensions
            mri_slice, ct_slice = [F.interpolate(tensor, size=(228, 228), mode='bilinear', align_corners=True) for tensor in [mri_slice, ct_slice]]
            
            # evaluate the current slice and store metrics
            metrics = self.evaluate_slice(mri_slice, ct_slice, patient_id, idx)
            metrics_list.append(metrics)
        
        return metrics_list

    def evaluate_dataset(self, root_dir):
        # evaluate the autoencoder model on the entire dataset
        all_metrics = []
        patient_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # iterate through each patient directory
        for patient in tqdm(patient_dirs, desc="Processing Patients"):
            mri_path, ct_path = os.path.join(root_dir, patient, 'mr.nii.gz'), os.path.join(root_dir, patient, 'ct.nii.gz')
            if os.path.exists(mri_path) and os.path.exists(ct_path):
                metrics = self.process_volume(mri_path, ct_path, patient)
                for m in metrics:
                    m['patient_id'], m['slice_position'] = patient, m['slice_idx'] / len(metrics)
                    all_metrics.append(m)
        
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(self.results_dir, 'metrics.csv'), index=False)
        self.plot_metrics(df)
        self.save_summary(df)

    def plot_metrics(self, df):
        metrics = ['psnr', 'ssim', 'mae', 'mse', 'nrmse', 'ncc', 'gradient_error', 'snr', 'cnr']
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        
        for ax, metric in zip(axes.flatten(), metrics):
            sns.boxplot(data=df, x='slice_position', y=metric, ax=ax)
            ax.set_title(f'{metric.upper()} Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics_distribution.png'))
        plt.close()

    def save_summary(self, df):
        # compute and save statistical summaries of evaluation metrics
        summary = {metric: df[metric].describe().to_dict() for metric in ['psnr', 'ssim', 'mae', 'mse', 'nrmse', 'ncc', 'gradient_error', 'snr', 'cnr']}
        with open(os.path.join(self.results_dir, 'summary.txt'), 'w') as f:
            for metric, stats in summary.items():
                f.write(f"{metric.upper()} Statistics:\n")
                for stat, val in stats.items():
                    f.write(f"{stat}: {val:.4f}\n")
                f.write("\n")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = load_model(device)
    evaluator = AutoencoderEvaluator(model, device, RESULTS_DIR)
    evaluator.evaluate_dataset(ROOT_DIR)
