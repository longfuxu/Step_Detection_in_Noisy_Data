import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generate step-like data with Gaussian noise using the provided function
def generate_step_data(n_points, step_locs, step_sizes, noise_std):
    x = np.linspace(0, 1, n_points)
    y = np.zeros(n_points)
    for loc, size in zip(step_locs, step_sizes):
        y += size * (x > loc)
    y += np.random.normal(0, noise_std, n_points)
    return x, y

# Fit a single step to the data
def fit_single_step(data):
    n_points = len(data)

    if n_points <= 5:
        return None, None, np.inf
    
    chi2 = np.zeros(n_points)
    for i in range(1, n_points):
        left_mean = np.mean(data[:i])
        right_mean = np.mean(data[i:])
        chi2[i] = np.sum((data[:i] - left_mean) ** 2) + np.sum((data[i:] - right_mean) ** 2)
    best_loc = np.argmin(chi2[1:-1]) + 1
    step_size = np.mean(data[best_loc:]) - np.mean(data[:best_loc])
    return best_loc, step_size, chi2[best_loc]

# Find steps in the data using the described algorithm
def find_steps(data, max_steps=100):
    step_locs = []
    step_sizes = []
    residuals = []
    remaining_data = np.copy(data)
    for _ in range(max_steps):
        best_loc, step_size, chi2 = fit_single_step(remaining_data)
        step_locs.append(best_loc)
        step_sizes.append(step_size)
        residuals.append(chi2)
        remaining_data[best_loc:] -= step_size
    return step_locs, step_sizes, residuals

# Modified find_steps function to find the optimal steps and step sizes based on step size threshold
def find_optimal_steps(data, max_steps=400, step_size_threshold=None):
    step_locs, step_sizes, residuals = find_steps(data, max_steps)
    
    # Sort the step sizes and corresponding step locations and residuals
    sorted_indices = np.argsort(step_sizes)[::-1]  # Sort from largest to smallest
    sorted_step_sizes = np.array(step_sizes)[sorted_indices]
    sorted_step_locs = np.array(step_locs)[sorted_indices]
    sorted_residuals = np.array(residuals)[sorted_indices]

    if step_size_threshold is not None:
        # Find the index where the step size is smaller than the step_size_threshold
        threshold_index = np.argmax(sorted_step_sizes < step_size_threshold)

        # Get the optimal step locations and sizes
        optimal_step_locs = sorted_step_locs[:threshold_index]
        optimal_step_sizes = sorted_step_sizes[:threshold_index]
    else:
        optimal_step_locs = sorted_step_locs
        optimal_step_sizes = sorted_step_sizes

    # Ensure step locations are unique and sorted
    unique_optimal_step_locs, unique_indices = np.unique(optimal_step_locs, return_index=True)
    unique_optimal_step_sizes = optimal_step_sizes[unique_indices]
    
    sorted_unique_indices = np.argsort(unique_optimal_step_locs)
    sorted_unique_step_locs = unique_optimal_step_locs[sorted_unique_indices]
    sorted_unique_step_sizes = unique_optimal_step_sizes[sorted_unique_indices]

    return sorted_unique_step_locs, sorted_unique_step_sizes, sorted_residuals

# Function to recalculate step sizes based on the mean values between adjacent step locations
def recalculate_step_sizes(data, step_locs):
    step_sizes = []
    n_steps = len(step_locs)
    for i in range(n_steps):
        if i == 0:
            left_data = data[:step_locs[i]]
        else:
            left_data = data[step_locs[i-1]:step_locs[i]]
        
        if i == n_steps - 1:
            right_data = data[step_locs[i]:]
        else:
            right_data = data[step_locs[i]:step_locs[i+1]]
        
        step_sizes.append(np.mean(right_data) - np.mean(left_data))
    return step_sizes

# Function to reconstruct the fitted curve
def reconstruct_fitted_curve(x, data, steps, step_sizes):
    fitted_curve = np.zeros_like(data)
    for step, step_size in zip(steps, step_sizes):
        fitted_curve[step:] += step_size
    return fitted_curve

def estimate_noise_std(data, scaling_factor=1.4826):
    # Calculate the difference between consecutive data points
    diff_data = np.diff(data)
    
    # Calculate the median absolute deviation (MAD) of the difference data
    mad = np.median(np.abs(diff_data - np.median(diff_data)))
    
    # Estimate the standard deviation using the scaling factor
    estimated_std = mad * scaling_factor
    return estimated_std

# Generate step-like traces
n_points = 1000
step_locs = [0.05, 0.1, 0.12, 0.15, 0.2, 0.21, 0.24, 0.3, 0.32, 0.4, 0.41, 0.52, 0.55, 0.6, 0.7, 0.77, 0.8, 0.81, 0.85, 0.9]
step_sizes = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
noise_std = 1.5
x, data = generate_step_data(n_points, step_locs, step_sizes, noise_std)

## Load data from the text file (replace 'data.txt' with the correct file name)
# data_imported = np.loadtxt('/Users/longfu/Desktop/testdata_regular_steps.txt')
# x = data_imported[:, 0]
# data = data_imported[:, 1]

# Apply the Savitzky-Golay filter
window_length = 5  # Choose an odd window length
polyorder = 3  # Choose the polynomial order for the local regression
filtered_data = savgol_filter(data, window_length, polyorder)

#scaling_factor ,should be in the range of [0.8,1.5], 
#is used to estimate the noise level, higher value give higher noise, lesss steps been detected.
estimated_noise_std = estimate_noise_std(filtered_data, scaling_factor=1.1) 

# Find the optimal steps and step sizes
optimal_step_locs, _, sorted_residuals = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std)

# Recalculate step sizes based on the optimal step locations
recalculated_step_sizes = recalculate_step_sizes(filtered_data, optimal_step_locs)

# Plot the origianl data
plt.figure()
plt.plot(x, data, label="Original Data")
plt.plot(x, filtered_data, label="Filtered Data")

# Reconstruct and plot the fitted curve
fitted_curve = reconstruct_fitted_curve(x, data, optimal_step_locs, recalculated_step_sizes)
plt.plot(x, fitted_curve, label="Fitted Steps")
plt.legend()

plt.figure()
plt.plot(range(len(sorted_residuals)), sorted_residuals, label="Sorted Residuals vs Iteration Steps")
plt.axvline(x=len(optimal_step_locs), color='b', linestyle='--', label="Threshold")
plt.legend()
plt.show()

print("Optimal step locations:", optimal_step_locs)
print("Recalculated step sizes:", recalculated_step_sizes)
print("Estimated noise standard deviation:", estimated_noise_std)
