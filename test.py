import numpy as np
from step_detection import generate_step_data, detect_steps

n_points = 1000
step_locs = [0.05, 0.1, 0.12, 0.15, 0.2, 0.21, 0.24, 0.3, 0.32, 0.4, 0.41, 0.52, 0.55, 0.6, 0.7, 0.77, 0.8, 0.81, 0.85, 0.9]
step_sizes = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
noise_std = 2
x, data = generate_step_data(n_points, step_locs, step_sizes, noise_std)

## Load data from the text file (replace 'data.txt' with the correct file name)
# data_imported = np.loadtxt('/Users/longfu/Desktop/your_data.txt')
# x = data_imported[:, 0]
# data = data_imported[:, 1]

# run the main function to detect steps and plot the original data, filtered data and fitted steps
"""
    1. x and y: These are the input data arrays, representing the x and y values of the data points.
    2. filter_window: This argument defines the window length of the Savitzky-Golay filter. 
    It should be an odd integer. Increasing the window length will result in smoother filtered data, 
    but it might also cause the loss of small, rapid changes in the data. 
    A smaller window length will retain more details in the data but might be more sensitive to noise. 
    Users should choose a window length that balances noise reduction and the preservation of relevant features in the data.
    3. filter_polyorder: This argument defines the polynomial order for the local regression used in the Savitzky-Golay filter. 
    A higher order will fit the data more accurately, but it might also capture more noise. 
    Users should choose a polynomial order that provides an appropriate balance between noise reduction and accurate fitting.
    4. scaling_factor: This value is used to estimate the noise level in the filtered data. 
    A larger scaling factor will result in a higher estimated noise level, which may lead to fewer steps being detected. 
    Conversely, a smaller scaling factor will result in a lower estimated noise level and potentially more steps being detected. 
    Users should choose a scaling factor that accurately represents the noise level in their data.
    5. distance_fraction: This fraction is used to calculate the minimum distance between adjacent steps based on the average distance between detected steps. 
    A larger distance fraction will require steps to be farther apart, which may result in fewer detected steps. 
    A smaller distance fraction will allow steps to be closer together, potentially detecting more steps. 
    Users should choose a distance_fraction that reflects the expected spacing between steps in their data.
"""
detect_steps(x, data,'Test_Data/',filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3)
