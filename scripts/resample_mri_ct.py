import os
import subprocess

# Paths
input_base = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/post/Task1/Pelvis"
output_base = "/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/processed"

# path to preprocessing script
preprocessing_script_path = "/Users/aoza/Mphy0043_cw/preprocessing/pre_process_tools.py"


# create the output directory if it doesn't exist
os.makedirs(output_base, exist_ok=True)

# loop through each patient folder in the input directory
for folder in os.listdir(input_base):
    patient_path = os.path.join(input_base, folder)

    # check if the current item is a folder
    if os.path.isdir(patient_path):  
        # Input file paths
        mr_input = os.path.join(patient_path, "mr.nii.gz")
        ct_input = os.path.join(patient_path, "ct.nii.gz")
        
        # Output file paths
        mr_output = os.path.join(output_base, f"{folder}_mr_resampled.nii.gz")
        ct_output = os.path.join(output_base, f"{folder}_ct_resampled.nii.gz")

        # skip processing if the output already exists or if folder is "1PA001"
        if os.path.exists(mr_output) or folder == "1PA001":
            print(f"Skipping {folder} as it is already processed.")
            continue

        # resample the MRI if the input file exists
        if os.path.exists(mr_input):
            try:
                subprocess.run([
                    "python", preprocessing_script_path, "resample",
                    "--i", mr_input,
                    "--o", mr_output,
                    "--s", "1", "1", "2.5"
                ], check=True)
                print(f"Processed MRI for {folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing MRI for {folder}: {e}")

        # resample the CT if the input file exists
        if os.path.exists(ct_input):
            try:
                subprocess.run([
                    "python", preprocessing_script_path, "resample",
                    "--i", ct_input,
                    "--o", ct_output,
                    "--s", "1", "1", "2.5"
                ], check=True)
                print(f"Processed CT for {folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing CT for {folder}: {e}")