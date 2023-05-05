import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

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

# Find steps in the data 
def find_steps(data, max_steps=400):
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
# and filter out steps that are too close to each other based on min_distance
def find_optimal_steps(data, max_steps=400, step_size_threshold=None, min_distance=None):
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

    # Filter out steps that are too close to each other based on min_distance
    if min_distance is not None:
        filtered_step_locs = [sorted_unique_step_locs[0]]
        filtered_step_sizes = [sorted_unique_step_sizes[0]]

        for i in range(1, len(sorted_unique_step_locs)):
            if sorted_unique_step_locs[i] - filtered_step_locs[-1] >= min_distance:
                filtered_step_locs.append(sorted_unique_step_locs[i])
                filtered_step_sizes.append(sorted_unique_step_sizes[i])

        sorted_unique_step_locs = np.array(filtered_step_locs)
        sorted_unique_step_sizes = np.array(filtered_step_sizes)

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

# To estimate the noise level of the input data
def estimate_noise_std(data, scaling_factor=1.4826):
    # Calculate the difference between consecutive data points
    diff_data = np.diff(data)
    
    # Calculate the median absolute deviation (MAD) of the difference data
    mad = np.median(np.abs(diff_data - np.median(diff_data)))
    
    # Estimate the standard deviation using the scaling factor
    estimated_std = mad * scaling_factor
    return estimated_std

# To estimate the min-distance between adjacent step locations, which should be larger than fraction * avg_distance
def estimate_min_distance(step_locs, fraction=0.3):
    # Calculate the differences between adjacent step locations
    step_differences = np.diff(step_locs)

    # Estimate the average distance between steps
    avg_distance = np.mean(step_differences)

    # Calculate the min_distance as a fraction of the average distance
    min_distance = int(fraction * avg_distance)

    return min_distance

# Main funtion to detect steps
def detect_steps(x, y, file_path,filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3):
    """
    0.file_path: the absolute path of your data files; used to save analyzed file in the same folder.
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
    # Apply the Savitzky-Golay filter
    filtered_data = savgol_filter(y, filter_window, filter_polyorder)

    # Estimate the noise standard deviation
    estimated_noise_std = estimate_noise_std(filtered_data, scaling_factor)

    # Find the optimal steps and step sizes
    optimal_step_locs, _, sorted_residuals = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std)
    min_distance = estimate_min_distance(optimal_step_locs, fraction=distance_fraction)
    optimal_step_locs, _, sorted_residuals = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std, min_distance=min_distance)

    # Recalculate step sizes based on the optimal step locations
    recalculated_step_sizes = recalculate_step_sizes(filtered_data, optimal_step_locs)

    # Reconstruct the fitted curve
    fitted_steps = reconstruct_fitted_curve(x, y, optimal_step_locs, recalculated_step_sizes)

    # Export original data, filtered data, and fitted data to a CSV file
    fitted_data_export = pd.DataFrame({
    "X": x[:len(filtered_data)],
    "Original Data": y[:len(filtered_data)],
    "Filtered Data": filtered_data,
    "Fitted Data": fitted_steps
    })
    # fitted_data_export.to_csv("/Users/longfu/Desktop/fitted_data_export.csv", index=False)

    # Export results of analyzed data to a CSV file
    result_export = pd.DataFrame({
    "Step Location": optimal_step_locs,
    "Step Size": recalculated_step_sizes
    })
    # result_export.to_csv("/Users/longfu/Desktop/result_export.csv", index=False)

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Plot the original data, filtered data, and fitted steps
    axes[0].plot(x, y, label="Original Data", linewidth=0.8)
    axes[0].plot(x, filtered_data, label="Filtered Data", linewidth=1)
    axes[0].plot(x, fitted_steps, label="Fitted Steps", linewidth=1.5)
    axes[0].legend()
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Basepairs Unwound')
    # Plot the quality check
    axes[1].plot(range(len(sorted_residuals)), sorted_residuals, label="Sorted Residuals vs Iteration Steps", linewidth=1, marker='o', markersize=4)
    axes[1].axvline(x=len(optimal_step_locs), color='b', linestyle='--', label="Threshold")
    axes[1].legend()
    axes[1].set_xlabel('Iteration Steps')
    axes[1].set_ylabel('Residuals')
    plt.tight_layout()
    # plt.savefig("/Users/longfu/Desktop/plot.svg", format='svg', dpi=300, bbox_inches='tight')
    
    # Extract directory and base name from the file_path
    file_dir = os.path.dirname(file_path)
    file_base_name, _ = os.path.splitext(os.path.basename(file_path))

    # Create new file names for the fitted data, result data, and the plot
    fitted_data_file = os.path.join(file_dir, f"{file_base_name}_fitted_data_export.csv")
    result_data_file = os.path.join(file_dir, f"{file_base_name}_result_export.csv")
    plot_file = os.path.join(file_dir, f"{file_base_name}_plot.eps")

    # Export original data, filtered data, and fitted data to a CSV file
    fitted_data_export.to_csv(fitted_data_file, index=False)

    # Export results of analyzed data to a CSV file
    result_export.to_csv(result_data_file, index=False)

    # Save the plot
    plt.savefig(plot_file, format='eps', dpi=300, bbox_inches='tight')

    # plt.show()

    print("min distance:", min_distance)
    print("Optimal step locations:", optimal_step_locs)
    print("Recalculated step sizes:", recalculated_step_sizes)
    print("Estimated noise standard deviation:", estimated_noise_std)
    return x, fitted_steps,optimal_step_locs, sorted_residuals

"""
# EXAMPLE USE CASE 
# Read data from the input file
file_path = "Example Data.csv"
data_imported = np.loadtxt(file_path, delimiter=',')
x = data_imported[:, 0]
data = data_imported[:, 1]

# Call the detect_steps function with the required arguments
detect_steps(x, data,filter_window=5, filter_polyorder=3, scaling_factor=1.1, distance_fraction=0.3)
"""
