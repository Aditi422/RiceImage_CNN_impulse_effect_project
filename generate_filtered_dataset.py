import cv2
import numpy as np
import os

# Function to add impulse noise to an image
def add_impulse_noise(image, p, salt_intensity=50):
    noisy_image = np.copy(image)
    mask = np.random.random(image.shape[:2])
    mask = np.clip(mask, 0, 1)

    for i in range(image.shape[2]):
        pepper_mask = mask < p / 2
        salt_mask = (p / 2 <= mask) & (mask < p)

        noisy_image[:, :, i][pepper_mask] = 0
        noisy_image[:, :, i][salt_mask] = salt_intensity

    return noisy_image

# Function to process a dataset with impulse noise
def process_dataset_impulse_noise(input_folder, output_folder, p, salt_intensity=50):
    os.makedirs(output_folder, exist_ok=True)
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)
        os.makedirs(output_class_path, exist_ok=True)

        for file_name in os.listdir(class_path):
            image_path = os.path.join(class_path, file_name)
            output_path = os.path.join(output_class_path, file_name)

            image = cv2.imread(image_path)
            noisy_image = add_impulse_noise(image, p, salt_intensity)
            cv2.imwrite(output_path, noisy_image)

# Example usage
input_dataset_folder = 'E:\thesis_aditiApu\Rice_Image_Dataset'
output_dataset_folder = 'E:\thesis_aditiApu\Filtered_Image_Dataset'

# Set the probability of impulse noise (p) and salt intensity
p = 0.02
salt_intensity = 50

process_dataset_impulse_noise(input_dataset_folder, output_dataset_folder, p, salt_intensity)
