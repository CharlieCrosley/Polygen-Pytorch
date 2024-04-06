import os
import shutil
import glob
from tqdm import tqdm

def copy_and_rename_models(parent_directory, destination_folder):
    model_counter = 0  # Counter to keep track of the new model name suffix
    
    # Automatically find all source folders starting with 'processed_'
    source_folders = glob.glob(os.path.join(parent_directory, 'processed_*'))

    # Filter out folders ending with '_objs'
    source_folders = [folder for folder in source_folders if not folder.endswith('_objs')]

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    for folder in source_folders:
        # Iterate through each folder in the source_folders list
        for subfolder in tqdm(os.listdir(folder)):
            subfolder_path = os.path.join(folder, subfolder)
            # Check if it is a directory and matches the model_X pattern
            if os.path.isdir(subfolder_path) and subfolder.startswith("model_"):
                # Define new model folder name based on the counter
                new_model_name = f"model_{model_counter}"
                # Path for the new model folder in the destination
                dest_path = os.path.join(destination_folder, new_model_name)
                # Copy the subfolder to the new location with the new name
                shutil.copytree(subfolder_path, dest_path)
                # Increment the counter for the next model's name
                model_counter += 1

# Parent directory containing source folders
parent_directory = "path/to/root/dir"
# Destination folder path
destination_folder = "path/to/dest"

# Call the function with the paths
copy_and_rename_models(parent_directory, destination_folder)
