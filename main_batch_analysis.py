import numpy as np
import os
import glob
from step_detection import detect_steps

# Set the root folder
root_folder = "/Users/longfu/Library/CloudStorage/Dropbox/5-T7 Helicase-SSB_PNAS/Raw data/Figure 1 and 2/Figure 1D"

# Find all CSV files in the root folder and its subfolders recursively
csv_files = glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True)

# Iterate through all CSV files
for file_path in csv_files:
    print(f"Processing {file_path}")

    # Read data from the input file
    data_imported = np.loadtxt(file_path, delimiter=',')
    x = data_imported[:, 0]
    data = data_imported[:, 1]

    # Call the detect_steps function with the required arguments
    detect_steps(x, data, file_path, filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3)
