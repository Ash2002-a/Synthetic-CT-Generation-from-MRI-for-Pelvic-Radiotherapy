# MRI-to-CT Translation Using Deep Learning Models

This project implements and compares different deep learning approaches for transforming MRI images into synthetic CT images, specifically focusing on Pix2Pix GAN and Autoencoder architectures. The implementation is designed for radiotherapy planning applications.

## Project Overview

The project includes implementations of:
- Pix2Pix GAN for MRI-to-CT translation
- Autoencoder baseline model
- Comprehensive evaluation scripts for both models
- Quantitative analysis using multiple metrics (MAE, MSE, PSNR, SSIM, SNR)


## Installation

1. Activate the course environment:
```bash
conda activate mphy0043
```

2. Install additional dependencies (if required):
```bash
pip install -r requirements.txt
```

## Project Structure

```
Mphy0043_cw/
├── scripts/
│   ├── pix2pix_model.py
│   ├── autoencoder.py
│   ├── evaluatePix2pix_model.py
│   └── evaluate_encoder.py
├── results/
├── autoencoder_output/
├── pix2pix_evaluation/
└── autoencoder_evaluation/
```

## Running 
### Pix2Pix Model Preparation

#### Dataset Preparation

1. **Download Dataset**
   - Download the dataset from: [https://zenodo.org/records/7260705] [Task 1]
   - Place the dataset in: `Mphy0043_cw/Data/mphy0043_cw/post`

2. **Convert NIfTI to PNG**
   ```bash
   cd Mphy0043_cw/scripts
   python convert_nifti_to_png.py
   ```
   - Processed images will be saved in: `Mphy0043_cw/Data/mphy0043_cw/processed`

### Pix2Pix Model

```bash
cd Mphy0043_cw/scripts
python pix2pix_model.py
```
Results will be saved in: `Mphy0043_cw/results/`

### Autoencoder Model

```bash
cd Mphy0043_cw/scripts
python autoencoder.py
```
Results will be saved in: `Mphy0043_cw/autoencoder_output/`

### Model Evaluation

For Pix2Pix:
```bash
cd Mphy0043_cw/scripts
python evaluate_Pix2pix.py
```
Results will be saved in: `Mphy0043_cw/pix2pix_evaluation/evaluation_[timestamp]/`

For Autoencoder:
```bash
cd Mphy0043_cw/scripts
python evaluate_encoder.py
```
Results will be saved in: `Mphy0043_cw/autoencoder_evaluation/`

## Output Directories

- `results/`: Contains Pix2Pix model outputs and training logs
- `autoencoder_output/`: Contains Autoencoder model outputs
- `pix2pix_evaluation/`: Contains evaluation metrics and visualisations for Pix2Pix
- `autoencoder_evaluation/`: Contains evaluation metrics and visualisations for Autoencoder

## Evaluation Metrics

The evaluation scripts calculate:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Signal-to-Noise Ratio (SNR)


# CycleGAN Pix2Pix Implementation Guide- Not used 

## Repository Setup
- Clone repository: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix]

## Preprocessing Scripts

### 1. Data Preparation
```bash
# Run preprocessing scripts in order
python resample_mri_ct.py
python split_data.py
python convert_to_png.py
```

### 2. Dataset Organization
- Create dataset directory: 
  `/Users/aoza/pytorch-CycleGAN-and-pix2pix/datasets/my_dataset_png`

## Model Training

### Train Pix2Pix Model
```bash
cd pytorch-CycleGAN-and-pix2pix
python train.py \
    --dataroot ./datasets/my_dataset_png \
    --name mri_to_ct_pix2pix \
    --model pix2pix \
    --direction AtoB \
    --gpu_ids -1
```

### Model Testing
```bash
python test.py \
    --dataroot /Users/aoza/pytorch-CycleGAN-and-pix2pix/datasets/test \
    --name mri_to_ct_pix2pix \
    --model pix2pix \
    --direction AtoB \
    --gpu_ids -1
```

## Post-Processing

### 1. Sort Real and Fake Images
```bash
python real_fake_sort.py
```

### 2. Model Evaluation
```bash
python evaluate_cycleGan.py
```

## Output Directories
```
pytorch-CycleGAN-and-pix2pix/
├── results/
│   └── mri_to_ct_pix2pix/
├── checkpoints/
│   └── mri_to_ct_pix2pix/
└── datasets/
    ├── my_dataset_png/
    └── test/
```

## Preprocessing tools (Git)
The following links were cloned and added to the project folder:

https://github.com/SynthRAD2023/preprocessing/tree/main
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## Dataset (Task 1):
https://zenodo.org/records/7260705


## End 
```
MPHY0043_CW/
│
├── autoencoder_evaluation/
│   └── evaluation_20250208_141808/
│       ├── comparisons/
│       │   └── metrics_distribution.png
│       ├── metrics.csv
│       └── summary.txt
│
├── pix2pix_evaluation/
│   └── evaluation_20250208_170722/
│       ├── comparisons/
│       │   └── metrics_over_epochs.png
│       ├── metrics.csv
│       └── summary.txt
│
├── Data/mphy0043_cw/
│   └── overview/
│       ├── post/
│       └── processed/
│
├── scripts/
│   ├── __pycache__/
│   ├── autoencoder.py
│   ├── convert_nifti_to_png.py
│   ├── convert_to_png.py
│   ├── eval_models.py
│   ├── eval_pix2.py
│   ├── evaluate_cycleGAN.py
│   ├── evaluate_encoder.py
│   ├── evaluate_pix2pix.py
│   ├── main.py
│   ├── pix2pix_model.py
│   ├── preprocess.py
│   ├── real_fake_sort.py
│   ├── resample_mri_ct.py
│   └── split_data.py
│
├── MPHY0043_Report.pdf
└── README.md
```