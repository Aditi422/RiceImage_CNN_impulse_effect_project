import os
from sklearn.model_selection import train_test_split
import shutil

def split_dataset(input_folder, output_train_folder, output_test_folder, test_size=0.2, random_state=42):
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_train_class_path = os.path.join(output_train_folder, class_folder)
        output_test_class_path = os.path.join(output_test_folder, class_folder)
        os.makedirs(output_train_class_path, exist_ok=True)
        os.makedirs(output_test_class_path, exist_ok=True)

        image_files = os.listdir(class_path)
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

        for file_name in train_files:
            input_path = os.path.join(class_path, file_name)
            output_path = os.path.join(output_train_class_path, file_name)
            shutil.copy(input_path, output_path)

        for file_name in test_files:
            input_path = os.path.join(class_path, file_name)
            output_path = os.path.join(output_test_class_path, file_name)
            shutil.copy(input_path, output_path)

# Example usage
input_dataset_folder = 'Rice_Image_Dataset'
output_train_dataset_folder = 'clean_train_dataset'
output_test_dataset_folder = 'clean_test_dataset'

split_dataset(input_dataset_folder, output_train_dataset_folder, output_test_dataset_folder, test_size=0.2)
