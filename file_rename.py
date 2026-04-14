import os

# List of target folders
folders = ["results_2", "results_5", "results_10", "results_20"]

for folder in folders:
    for filename in os.listdir(folder):
        if "PALM" in filename:
            old_path = os.path.join(folder, filename)
            
            # Replace "PALM" with "DFM" in filename
            new_filename = filename.replace("PALM", "DFM")
            new_path = os.path.join(folder, new_filename)
            
            # Rename file
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
