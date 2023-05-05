# Step Detection in Noisy Data

---
All codes listed in this repository are developed by Longfu Xu during the phD work in Gijs Wuite Group, to investigate the DNA replication at the single-molecule level.

Please note that the code in this repository is custom written for internal lab use and still may contain bugs; external support cannot be guaranteed.

Developer: Longfu Xu (longfuxu.com) . Maintenance, development, support. For questions or reports, e-mail: l2.xu@vu.nl
---

This Python script provides a method for detecting step locations and sizes in step-like traces with Gaussian noise. The method involves generating a step-like trace, estimating the noise level, filtering the noise, and finally detecting step locations and sizes.

## Table of Contents

- Introduction
- Usage
- Example
- Detailed Outline

## Introduction

Detecting steps in noisy data is a common problem in various scientific fields. This script aims to provide an efficient and accurate way to detect steps in the presence of Gaussian noise. The method consists of generating a step-like trace, estimating the noise level, filtering the noise, and detecting the step locations and sizes.

## Usage

1. Import the required libraries:

```
pythonCopy code
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
```

1. Define the necessary functions:

- `generate_step_data`: Generates a step-like dataset with Gaussian noise.
- `estimate_noise_std`: Estimates the noise level in the data.
- `fit_single_step`: Fits a single step to the data and calculates the residuals.
- `find_optimal_steps`: Finds the optimal step locations and sizes.
- `reconstruct_fitted_curve`: Reconstructs the fitted curve based on the optimal step locations and sizes.

1. Generate the step-like trace, filter the noise, and detect the steps:

- Define the number of points, step locations, step sizes, and noise standard deviation.
- Estimate the noise level in the data.
- Filter the noise using the Savitzky-Golay filter.
- Find the optimal step locations and sizes using the `find_optimal_steps` function.

1. Visualize the results:

- Plot the original data, the filtered data, and the fitted curve.
- Display the optimal step locations and recalculated step sizes.

## Example

```
n_points = 4000
step_locs = [0.05, 0.1, 0.12, 0.15, 0.2, 0.21, 0.24, 0.3, 0.32, 0.4, 0.41, 0.52, 0.55, 0.6, 0.7, 0.77, 0.8, 0.81, 0.85, 0.9]
step_sizes = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
noise_std = 1.5
```

## Detailed Outline

1. **Generate a step-like trace with Gaussian noise:**

   - Define the number of points, step locations, step sizes, and noise standard deviation.
   - Use the `generate_step_data` function to create a simulated dataset with the given parameters.

2. **Estimate the noise level:**

   - Estimate the standard deviation of the Gaussian noise present in the data using the `estimate_noise_std` function.
   - This function calculates the median absolute deviation (MAD) of the differences between consecutive data points and scales it with a factor (e.g., 1.4826) to estimate the standard deviation.

3. **Filter the noise:**

   - Apply the Savitzky-Golay filter to the data using the `savgol_filter` function from the `scipy` library.

   - Choose an appropriate window length (odd integer) and polynomial order for 

   - the filter, ensuring that the window length is large enough to smooth the noise but small enough to preserve the step features.

     1. **Detect step locations and sizes:**
        - Use the `fit_single_step` function to fit a single step to the filtered data, calculate the residuals, and find the best location for the step.
        - Apply the `find_optimal_steps` function to iteratively find the optimal step locations and sizes based on the calculated residuals.
        - Define a minimum step size threshold, typically equal to the estimated noise standard deviation, and filter out the steps with sizes smaller than this threshold.
        - Recalculate the step sizes by taking the mean difference between adjacent optimal step locations.
     2. **Reconstruct the fitted curve:**
        - Use the `reconstruct_fitted_curve` function to reconstruct the fitted curve based on the optimal step locations and recalculated step sizes.
        - This function iteratively adds the recalculated step sizes to the fitted curve, starting from the optimal step locations.
     3. **Visualize the results:**
        - Plot the original data, the filtered data, and the reconstructed fitted curve.
        - Display the optimal step locations and recalculated step sizes in a clear and concise manner.
        - You can customize the plots (e.g., colors, labels, markers) to make the visualization more informative and appealing.

     Here's an example of how the final script may look like:

     ```
     import numpy as np
     import matplotlib.pyplot as plt
     from scipy.signal import savgol_filter
     
     # Define required functions and generate data (as shown previously)
     
     # Filter the data
     window_length = 51
     poly_order = 2
     filtered_data = savgol_filter(data, window_length, poly_order)
     
     # Find the optimal step locations and sizes
     optimal_step_locs, optimal_step_sizes, sorted_indices = find_optimal_steps(filtered_data)
     
     # Recalculate the step sizes
     recalculated_step_sizes = calculate_step_sizes(data, optimal_step_locs)
     
     # Reconstruct the fitted curve
     fitted_curve = reconstruct_fitted_curve(x, data, optimal_step_locs, recalculated_step_sizes)
     
     # Plot the results
     plt.figure()
     plt.plot(x, data, label="Original Data")
     plt.plot(x, filtered_data, label="Filtered Data")
     plt.plot(x, fitted_curve, linestyle='--', color="r", label="Fitted Curve")
     plt.legend()
     
     # Display the optimal step locations and recalculated step sizes
     print("Optimal step locations:", optimal_step_locs)
     print("Recalculated step sizes:", recalculated_step_sizes)
     ```
     
By following the instructions in this README, users should be able to understand the theory behind the step detection method, and know how to apply the provided script to their own datasets.