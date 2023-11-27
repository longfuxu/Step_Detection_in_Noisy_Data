# Step Detection in Noisy Data

This Python script is designed to identify steps in noisy data, specifically Gaussian noise. The script first filters the noise to reveal the underlying step-like structure, and then it applies a step-fitting algorithm to detect the locations and sizes of the steps.

## Overview

1. **Data Generation**: The script starts by generating synthetic step-like data mixed with Gaussian noise. This can be replaced with real data if necessary.

2. **Noise Filtering**: A Savitzky-Golay filter is applied to the noisy data to smooth out the noise and reveal the underlying step-like structure.

3. **Noise Estimation**: The script estimates the standard deviation of the noise in the original data using the median absolute deviation (MAD) method.

4. **Step Detection**: A custom step-detection algorithm is applied to the filtered data to identify the locations and sizes of the steps. The algorithm iteratively fits single steps to the data and subtracts them, refining the fit until a specified threshold is reached. The threshold is based on the estimated noise level.

## Algorithm Details

### Noise Filtering: Savitzky-Golay Filter

The Savitzky-Golay filter is used to smooth the noisy data. It fits a polynomial of a specified order to a moving window of data points and uses the fitted polynomial to replace the central point in the window. This process is repeated for all data points, resulting in a smoothed version of the input data.

Parameters for the Savitzky-Golay filter:

- `window_length`: The size of the moving window used for the local polynomial regression. Must be an odd integer.
- `polyorder`: The order of the polynomial used for the local regression.

### Noise Estimation: Median Absolute Deviation (MAD)

The MAD method is used to estimate the standard deviation of the Gaussian noise in the input data. The script calculates the median absolute deviation of the differences between consecutive data points and multiplies it by a scaling factor (approximately 1.4826) to estimate the standard deviation of the noise.

### Step Detection: Custom Step-Fitting Algorithm

The step detection algorithm iteratively fits single steps to the data and subtracts them, refining the fit until a specified threshold is reached. The threshold is based on the estimated noise level.

The algorithm first fits a single step to the data and calculates the residuals. Then, it sorts the residuals and selects the largest ones that account for 95% of the initial total residual. The step locations and sizes are determined based on these selected residuals. Finally, the algorithm combines the optimal steps and step sizes based on the indices of the sorted arrays.

## Usage

Simply run the Python script to generate synthetic step-like data with Gaussian noise, apply the Savitzky-Golay filter to smooth the data, estimate the noise level, and detect the steps using the custom step-fitting algorithm. The script will output the step locations and sizes, as well as the estimated noise level.