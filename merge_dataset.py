import os
import shutil

def merge_datasets(noisy20_train_dataset_folder, 
                   noisy40_train_dataset_folder, 
                   noisy60_train_dataset_folder, 
                   merged_probability_train_dataset_folder):
    os.makedirs(merged_probability_train_dataset_folder, exist_ok=True)

    for class_folder in os.listdir(noisy20_train_dataset_folder):
        noisy20_class_path = os.path.join(noisy20_train_dataset_folder, class_folder)
        noisy40_class_path = os.path.join(noisy40_train_dataset_folder, class_folder)
        noisy60_class_path = os.path.join(noisy60_train_dataset_folder, class_folder)

        merged_dataset_folder = os.path.join(merged_probability_train_dataset_folder, class_folder)
        os.makedirs(merged_dataset_folder, exist_ok=True)

        # Copy noisy20 images to merged folder
        noisy20_files = os.listdir(noisy20_class_path)
        for i, file_name in enumerate(noisy20_files):
            noisy20_image_path = os.path.join(noisy20_class_path, file_name)
            merged_image_path = os.path.join(merged_dataset_folder, f'noisy20_{i}_{file_name}')
            shutil.copy(noisy20_image_path, merged_image_path)

        # Copy noisy40 images to merged folder
        noisy40_files = os.listdir(noisy40_class_path)
        for i, file_name in enumerate(noisy40_files):
            noisy40_image_path = os.path.join(noisy40_class_path, file_name)
            merged_image_path = os.path.join(merged_dataset_folder, f'noisy40_{i}_{file_name}')
            shutil.copy(noisy40_image_path, merged_image_path)

        # Copy noisy60 images to merged folder
        noisy60_files = os.listdir(noisy60_class_path)
        for i, file_name in enumerate(noisy60_files):
            noisy60_image_path = os.path.join(noisy60_class_path, file_name)
            merged_image_path = os.path.join(merged_dataset_folder, f'noisy60_{i}_{file_name}')
            shutil.copy(noisy60_image_path, merged_image_path)

# Example usage
noisy20_train_dataset_folder = '20_noise_train_dataset'
noisy40_train_dataset_folder = '40_noise_train_dataset'
noisy60_train_dataset_folder = '60_noise_train_dataset'
merged_probability_train_dataset_folder = 'probability_train_dataset'

merge_datasets(noisy20_train_dataset_folder, 
               noisy40_train_dataset_folder, 
               noisy60_train_dataset_folder, 
               merged_probability_train_dataset_folder)
