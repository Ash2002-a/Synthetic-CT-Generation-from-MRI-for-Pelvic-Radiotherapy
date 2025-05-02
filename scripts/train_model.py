import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import nibabel as nib
from sklearn.model_selection import train_test_split
import time

# directory paths setup
base_dir = '/Users/aoza/Mphy0043_cw/'
data_dir = '/Users/aoza/Mphy0043_cw/Data/mphy0043_cw/post/Task1/Pelvis/'  # Correct path to processed folder
models_dir = os.path.join(base_dir, 'models/')
results_dir = os.path.join(base_dir, 'results/')

# check if directories exist, create them if not
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# generator architecture definition
def build_generator():
    def residual_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Add()([shortcut, x])

    input_img = layers.Input(shape=(256, 256, 1))
    
    # initial convolution layer
    x = layers.Conv2D(64, 7, padding='same', use_bias=False)(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # downsampling layers
    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # add residual blocks
    for _ in range(9):
        x = residual_block(x, 256)
    
    # upsampling layers
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # final convolution for output
    x = layers.Conv2D(1, 7, padding='same', activation='tanh')(x)
    
    return models.Model(input_img, x)

# discriminator architecture definition
def build_discriminator():
    input_img = layers.Input(shape=(256, 256, 1))
    
    x = layers.Conv2D(64, 4, strides=2, padding='same')(input_img)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(1, 4, padding='same')(x)
    
    return models.Model(input_img, x)

# CycleGAN class definition
class CycleGAN:
    def __init__(self):
        print("Initializing CycleGAN...")
        # create generators and discriminators
        self.g_mri2ct = build_generator()
        self.g_ct2mri = build_generator()
        self.d_mri = build_discriminator()
        self.d_ct = build_discriminator()
        
        # define optimizers for the model
        self.generator_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
        
        # set loss weights for cycle consistency and identity loss
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5
        print("CycleGAN initialized.")

    # generator loss function
    def generator_loss(self, generated_output):
        return tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(tf.ones_like(generated_output), generated_output))
    
    # discriminator loss function
    def discriminator_loss(self, real_output, generated_output):
        real_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(tf.ones_like(real_output), real_output))
        generated_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(tf.zeros_like(generated_output), generated_output))
        return (real_loss + generated_loss) * 0.5
    
    # cycle consistency loss function
    def cycle_loss(self, real_image, cycled_image):
        return tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    # identity loss function
    def identity_loss(self, real_image, same_image):
        return tf.reduce_mean(tf.abs(real_image - same_image))
    
    # training step for each batch
    @tf.function
    def train_step(self, mri, ct):
        with tf.GradientTape(persistent=True) as tape:
            # generate images from the input images
            fake_ct = self.g_mri2ct(mri, training=True)
            fake_mri = self.g_ct2mri(ct, training=True)
            
            # cycle images through both generators
            cycled_mri = self.g_ct2mri(fake_ct, training=True)
            cycled_ct = self.g_mri2ct(fake_mri, training=True)
            
            # apply identity mapping to test the generators
            same_mri = self.g_ct2mri(mri, training=True)
            same_ct = self.g_mri2ct(ct, training=True)
            
            # discriminator outputs for real and fake images
            disc_real_mri = self.d_mri(mri, training=True)
            disc_real_ct = self.d_ct(ct, training=True)
            disc_fake_mri = self.d_mri(fake_mri, training=True)
            disc_fake_ct = self.d_ct(fake_ct, training=True)
            
            # calculate generator losses
            gen_mri2ct_loss = self.generator_loss(disc_fake_ct)
            gen_ct2mri_loss = self.generator_loss(disc_fake_mri)
            
            #calculate cycle consistency losses
            cycle_mri_loss = self.cycle_loss(mri, cycled_mri)
            cycle_ct_loss = self.cycle_loss(ct, cycled_ct)
            total_cycle_loss = cycle_mri_loss + cycle_ct_loss
            
            # calculate identity losses
            identity_mri_loss = self.identity_loss(mri, same_mri)
            identity_ct_loss = self.identity_loss(ct, same_ct)
            
            # total generator losses
            total_gen_mri2ct_loss = (gen_mri2ct_loss + 
                                   total_cycle_loss * self.lambda_cycle +
                                   identity_ct_loss * self.lambda_identity)
            total_gen_ct2mri_loss = (gen_ct2mri_loss + 
                                   total_cycle_loss * self.lambda_cycle +
                                   identity_mri_loss * self.lambda_identity)
            
            # discriminator losses
            disc_mri_loss = self.discriminator_loss(disc_real_mri, disc_fake_mri)
            disc_ct_loss = self.discriminator_loss(disc_real_ct, disc_fake_ct)
        
        # calculate gradients and apply them
        gen_mri2ct_gradients = tape.gradient(total_gen_mri2ct_loss, 
                                           self.g_mri2ct.trainable_variables)
        gen_ct2mri_gradients = tape.gradient(total_gen_ct2mri_loss, 
                                           self.g_ct2mri.trainable_variables)
        disc_mri_gradients = tape.gradient(disc_mri_loss, 
                                         self.d_mri.trainable_variables)
        disc_ct_gradients = tape.gradient(disc_ct_loss, 
                                        self.d_ct.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gen_mri2ct_gradients, self.g_mri2ct.trainable_variables))
        self.generator_optimizer.apply_gradients(
            zip(gen_ct2mri_gradients, self.g_ct2mri.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(disc_mri_gradients, self.d_mri.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(disc_ct_gradients, self.d_ct.trainable_variables))
        
        return {
            'gen_mri2ct_loss': total_gen_mri2ct_loss,
            'gen_ct2mri_loss': total_gen_ct2mri_loss,
            'disc_mri_loss': disc_mri_loss,
            'disc_ct_loss': disc_ct_loss
        }

# data loading and preprocessing function
def load_and_preprocess_data(data_dir, target_shape=(256, 256)):
    mri_images = []
    ct_images = []
    
    for patient_folder in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient_folder)
        if os.path.isdir(patient_path):
            mri_path = os.path.join(patient_path, 'mr.nii.gz')
            ct_path = os.path.join(patient_path, 'ct.nii.gz')
            
            if os.path.exists(mri_path) and os.path.exists(ct_path):
                print(f"Loading MRI from {mri_path}")
                mri_img = nib.load(mri_path).get_fdata()
                print(f"Loading CT from {ct_path}")
                ct_img = nib.load(ct_path).get_fdata()
                
                # process slices from the volume
                for slice_idx in range(mri_img.shape[2]):
                    mri_slice = mri_img[:, :, slice_idx]
                    ct_slice = ct_img[:, :, slice_idx]
                    
                    # resize
                    mri_slice = tf.image.resize(mri_slice[..., None], target_shape)
                    ct_slice = tf.image.resize(ct_slice[..., None], target_shape)
                    
                    # normalize to [-1, 1]
                    mri_slice = (mri_slice - tf.reduce_min(mri_slice)) / (tf.reduce_max(mri_slice) - tf.reduce_min(mri_slice))
                    ct_slice = (ct_slice - tf.reduce_min(ct_slice)) / (tf.reduce_max(ct_slice) - tf.reduce_min(ct_slice))
                    mri_slice = mri_slice * 2 - 1
                    ct_slice = ct_slice * 2 - 1
                    
                    mri_images.append(mri_slice)
                    ct_images.append(ct_slice)
    
    return np.array(mri_images), np.array(ct_images)

# initialize logging file
training_log_path = os.path.join(results_dir, 'training_log.csv')
with open(training_log_path, 'w') as f:
    f.write("Epoch,Batch,Gen MRI->CT Loss,Gen CT->MRI Loss,Disc MRI Loss,Disc CT Loss\n")

# training function
def train_cyclegan(data_dir, epochs=100, batch_size=1):
    print("Starting CycleGAN training...")
    # initialize model
    cyclegan = CycleGAN()
    
    # load and preprocess data
    print("Loading and preprocessing data...")
    mri_images, ct_images = load_and_preprocess_data(data_dir)
    print(f"Loaded {len(mri_images)} MRI slices and {len(ct_images)} CT slices.")
    
    # split the data into training, validation, and testing sets
    mri_train, mri_temp, ct_train, ct_temp = train_test_split(mri_images, ct_images, test_size=0.4, random_state=42)
    mri_val, mri_test, ct_val, ct_test = train_test_split(mri_temp, ct_temp, test_size=0.5, random_state=42)
    
    print(f"Training MRI images: {mri_train.shape}")
    print(f"Validation MRI images: {mri_val.shape}")
    print(f"Testing MRI images: {mri_test.shape}")
    print(f"Training CT images: {ct_train.shape}")
    print(f"Validation CT images: {ct_val.shape}")
    print(f"Testing CT images: {ct_test.shape}")
    
    # create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((mri_train, ct_train)).shuffle(len(mri_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((mri_val, ct_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((mri_test, ct_test)).batch(batch_size)
    
    # initialize training history
    history = {
        'gen_mri2ct_loss': [],
        'gen_ct2mri_loss': [],
        'disc_mri_loss': [],
        'disc_ct_loss': []
    }
    
    # training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        num_batches = len(mri_train) // batch_size
        print(f"Total number of batches: {num_batches}")
        
        start_time = time.time()
        
        for batch_idx, (mri_batch, ct_batch) in enumerate(train_dataset):
            losses = cyclegan.train_step(mri_batch, ct_batch)
            
            # log losses after every batch
            with open(training_log_path, 'a') as f:
                f.write(f"{epoch+1},{batch_idx},{losses['gen_mri2ct_loss']:.4f},{losses['gen_ct2mri_loss']:.4f},{losses['disc_mri_loss']:.4f},{losses['disc_ct_loss']:.4f}\n")
            
            if batch_idx % 50 == 0:
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / (batch_idx + 1)) * (num_batches - batch_idx - 1)
                print(f"Batch {batch_idx}: Gen MRI->CT loss: {losses['gen_mri2ct_loss']:.4f}, Gen CT->MRI loss: {losses['gen_ct2mri_loss']:.4f}, ETA: {eta:.2f} seconds")
        
        # save models and loss metrics periodically
        if (epoch + 1) % 10 == 0:
            print(f"Saving models and loss metrics for epoch {epoch + 1}...")
            cyclegan.g_mri2ct.save(os.path.join(models_dir, f'generator_mri2ct_epoch_{epoch+1}.h5'))
            cyclegan.g_ct2mri.save(os.path.join(models_dir, f'generator_ct2mri_epoch_{epoch+1}.h5'))
            
            # save detailed loss metrics
            loss_metrics_path = os.path.join(results_dir, f'loss_metrics_epoch_{epoch+1}.txt')
            with open(loss_metrics_path, 'w') as f:
                f.write(f"Cycle Loss MRI to CT: {losses['cycle_mri_loss'].numpy()}\n")
                f.write(f"Cycle Loss CT to MRI: {losses['cycle_ct_loss'].numpy()}\n")
                f.write(f"Identity Loss MRI: {losses['identity_mri_loss'].numpy()}\n")
                f.write(f"Identity Loss CT: {losses['identity_ct_loss'].numpy()}\n")
                f.write(f"Discriminator MRI Loss: {losses['disc_mri_loss'].numpy()}\n")
                f.write(f"Discriminator CT Loss: {losses['disc_ct_loss'].numpy()}\n")
        
        # append losses to history
        history['gen_mri2ct_loss'].append(losses['gen_mri2ct_loss'].numpy())
        history['gen_ct2mri_loss'].append(losses['gen_ct2mri_loss'].numpy())
        history['disc_mri_loss'].append(losses['disc_mri_loss'].numpy())
        history['disc_ct_loss'].append(losses['disc_ct_loss'].numpy())

    print("CycleGAN training completed.")
    return cyclegan, history


# xall the training function
cyclegan, training_history = train_cyclegan(data_dir, epochs=100, batch_size=1)

# print the training history
print("Training history:")
print(training_history)