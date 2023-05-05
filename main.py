import numpy as np
from step_detection import detect_steps

# Read data from the input file
file_path = "/Users/longfu/Library/CloudStorage/Dropbox/5-T7 Helicase-SSB_PNAS/Raw data/Figure 1 and 2/Figure 2A/gp4+gp2.5/gp4 + gp2.5 -45pN-2.csv"
data_imported = np.loadtxt(file_path, delimiter=',')
x = data_imported[:, 0]
data = data_imported[:, 1]

# Call the detect_steps function with the required arguments
detect_steps(x, data,file_path, filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3)

# # If you want to process more files, you can use a loop and call the detect_steps function for each file
# file_paths = ["file1.csv", "file2.csv", "file3.csv"]
# for file_path in file_paths:
#     data_imported = np.loadtxt(file_path, delimiter=',')
#     x = data_imported[:, 0]
#     y = data_imported[:, 1]
#     detect_steps(x, data,filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3)


# detect_steps(x, data,filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3)