import os
import shutil

def merge_datasets(clean_dataset_folder, noisy_dataset_folder, merged_dataset_folder):
    os.makedirs(merged_dataset_folder, exist_ok=True)

    for class_folder in os.listdir(clean_dataset_folder):
        clean_class_path = os.path.join(clean_dataset_folder, class_folder)
        noisy_class_path = os.path.join(noisy_dataset_folder, class_folder)
        merged_class_path = os.path.join(merged_dataset_folder, class_folder)
        os.makedirs(merged_class_path, exist_ok=True)

        # Copy clean images to merged folder
        clean_files = os.listdir(clean_class_path)
        for i, file_name in enumerate(clean_files):
            clean_image_path = os.path.join(clean_class_path, file_name)
            merged_image_path = os.path.join(merged_class_path, f'clean_{i}_{file_name}')
            shutil.copy(clean_image_path, merged_image_path)

        # Copy noisy images to merged folder
        noisy_files = os.listdir(noisy_class_path)
        for i, file_name in enumerate(noisy_files):
            noisy_image_path = os.path.join(noisy_class_path, file_name)
            merged_image_path = os.path.join(merged_class_path, f'noisy_{i}_{file_name}')
            shutil.copy(noisy_image_path, merged_image_path)

# Example usage
clean_train_dataset_folder = 'clean_train_dataset'
noisy_train_dataset_folder = 'noisy_train_dataset'
merged_train_dataset_folder = 'train_dataset'

merge_datasets(clean_train_dataset_folder, noisy_train_dataset_folder, merged_train_dataset_folder)
