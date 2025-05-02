import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_images(mri_dir, image_size=(256, 256)):
    """
    Load and preprocess MRI images from the specified directory.
    Resize the images to the target size and normalize them.

    Args:
    - mri_dir (str): Directory containing the MRI images.
    - image_size (tuple): The target size for resizing images (default is 256x256).
    
    Returns:
    - np.array: Preprocessed images (normalized and resized).
    """
    images = [] # initialize an empty list to hold the images

    # loop through the files in the specified directory
    for image_name in os.listdir(mri_dir):
        if image_name.endswith('.png'):
            img_path = os.path.join(mri_dir, image_name)

            # load the image, resize it, and convert to grayscale (if not already)
            img = load_img(img_path, target_size=image_size, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0  # normalize the image to [0, 1]
            images.append(img_array)
    
    return np.array(images)


mri_images = load_and_preprocess_images("/path/to/mri/images")
mri_images = np.expand_dims(mri_images, axis=-1)  # add the channel dimension if needed
print(f"Preprocessed MRI Images Shape: {mri_images.shape}")
