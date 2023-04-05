# Step Detection in Noisy Data

**Introduction**

In many applications, we are interested in analyzing step-like behavior in data where sudden changes occur. Such data can be affected by Gaussian noise, making it challenging to identify the exact locations and magnitudes of these steps. This document presents a method for generating simulated step-like data, estimating the noise level, filtering the noise, and detecting the step locations and sizes in the data. We also provide a discussion on the underlying theory behind each step of the process.

**Method**

1. **Generate a step-like trace with Gaussian noise:**

   The first step in the process is to generate a simulated dataset that consists of a step-like signal with steps of varying size and duration, hidden in Gaussian noise with a root mean square (RMS) amplitude (σ). The `generate_step_data` function is used for this purpose, taking the following parameters:

   - `n_points`: The number of data points in the dataset.
   - `step_locs`: A list of the relative positions of the steps in the dataset.
   - `step_sizes`: A list of the sizes (magnitudes) of the steps.
   - `noise_std`: The standard deviation of the Gaussian noise.

   The function creates a step-like signal by adding the step sizes at the specified locations and then adds Gaussian noise with the given standard deviation.

   Example:

   ```python
   n_points = 4000
   step_locs = [0.05, 0.1, 0.12, 0.15, 0.2, 0.21, 0.24, 0.3, 0.32, 0.4, 0.41, 0.52, 0.55, 0.6, 0.7, 0.77, 0.8, 0.81, 0.85, 0.9]
   step_sizes = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
   noise_std = 1.5
   x, data = generate_step_data(n_points, step_locs, step_sizes, noise_std)
   ```
   
2. **Estimate the noise level:**

   In real-world applications, the standard deviation of the Gaussian noise may not be known. The `estimate_noise_std` function estimates the noise level by calculating the median absolute deviation (MAD) of the differences between consecutive data points and scaling it with a factor (e.g., 1.4826) to obtain an unbiased estimate of the standard deviation.

   Example:

   ```python
   estimated_noise_std = estimate_noise_std(data)
   ```
   
3. **Filter the noise:**

   To reduce the impact of noise on the step detection process, we apply the Savitzky-Golay filter to the data. This filter smooths the data by fitting a polynomial of a specified order to a sliding window of data points and replaces the central data point with the fitted value. The `savgol_filter` function from the `scipy` library is used for this purpose, with the following parameters:

   - `window_length`: The length of the sliding window (must be an odd integer).
   - `polyorder`: The order of the polynomial used in the local regression.

   Example:

   ```
   from scipy.signal import savgol_filter
   
   window_length = 51
   polyorder = 3
   filtered_data = savgol_filter(data, window_length, polyorder)
   ```
   
4. **Find the step locations and sizes:**

   The core of the method is identifying the step locations and sizes in the filtered data. We use the `find_optimal_steps` function, which iteratively fits single steps to the data and calculates the residuals between the data and the fitted steps. The function sorts the steps by their sizes and filters out the steps that are smaller than a threshold (e.g., the estimated noise standard deviation). The optimal steps and step sizes are then combined based on the indices of the original sorted arrays.

   Example:

   ```python
   optimal_step_locs, optimal_step_sizes = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std)
   ```
   
5. **Reconstruct the fitted curve:**

   To visualize the results, we can reconstruct the fitted curve using the detected step locations and sizes. The `reconstruct_fitted_curve` function is used for this purpose, taking the following parameters:

   - `x`: The array of x values (independent variable).
   - `data`: The original noisy data.
   - `optimal_step_locs`: The detected optimal step locations.
   - `optimal_step_sizes`: The detected optimal step sizes.

   The function creates a new array with the same shape as the input data and adds the detected step sizes at the optimal step locations.

   Example:

   ```
   fitted_curve = reconstruct_fitted_curve(x, data, optimal_step_locs, optimal_step_sizes)
   ```
   
6. **Plot the results:**

   Finally, we can plot the original data, filtered data, and the fitted curve to visualize the effectiveness of the method.

   Example:

   ```
   import matplotlib.pyplot as plt
   
   plt.figure()
   plt.plot(x, data, label="Original Data")
   plt.plot(x, filtered_data, label="Filtered Data")
   plt.plot(x, fitted_curve, linestyle='--', color="r", label="Fitted Curve")
   plt.legend()
   plt.show()
   ```

**Discussion**

The method presented in this document provides a systematic approach to detecting step-like behavior in data affected by Gaussian noise. The use of a noise estimator, a noise filter, and an iterative step-fitting algorithm allows for accurate detection of step locations and sizes, even in cases where the noise level is high or unknown.

However, the method's performance can be affected by the choice of filter parameters, such as the window length and polynomial order in the Savitzky-Golay filter. These parameters should be chosen carefully to balance noise reduction and preserving the original signal's features. Additionally, the method assumes that the noise is Gaussian, which may not always be the case in real-world data.

Despite these limitations, this method provides a robust starting point for detecting steps in noisy data and can be further refined and adapted to specific applications as needed.