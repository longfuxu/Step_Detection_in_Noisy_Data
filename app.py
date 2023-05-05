import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from step_detection import detect_steps, generate_step_data

def main():
    st.title("Step Detection in Noisy Data")

    st.sidebar.markdown("## Import Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt', 'csv'])

    use_simulated_data = st.sidebar.button("Try A Simulated Data")

    if 'x' not in st.session_state:
        st.session_state.x = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    
    if uploaded_file:
        st.session_state.file_path = uploaded_file.name

    if uploaded_file:
        data_imported = np.loadtxt(uploaded_file, delimiter=',') # make sure your dataset has a correct format when import
        # data_imported = np.loadtxt(uploaded_file)
        st.session_state.x = data_imported[:, 0]
        st.session_state.data = data_imported[:, 1]
        st.write("Data loaded:")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.x, st.session_state.data)
        ax.set_xlabel("X-axis label")
        ax.set_ylabel("Y-axis label")
        st.pyplot(fig)

    elif use_simulated_data:
        n_points = 1000
        step_locs = [0.05, 0.1, 0.12, 0.15, 0.2, 0.21, 0.24, 0.3, 0.32, 0.4, 0.41, 0.52, 0.55, 0.6, 0.7, 0.77, 0.8, 0.81, 0.85, 0.9]
        step_sizes = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        noise_std = 0.3
        st.session_state.x, st.session_state.data = generate_step_data(n_points, step_locs, step_sizes, noise_std)
        st.write("Simulated data loaded:")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.x, st.session_state.data)
        ax.set_xlabel("X-axis label")
        ax.set_ylabel("Y-axis label")
        st.pyplot(fig)

    st.sidebar.markdown("## Parameters")
    filter_window = int(st.sidebar.number_input("Filter Window", min_value=5, value=5, step=1))
    filter_polyorder = int(st.sidebar.number_input("Filter Polynomial Order", min_value=3, value=3, step=1))
    scaling_factor = st.sidebar.number_input("Scaling Factor", min_value=0.8, value=1.1, step=0.1)
    distance_fraction = st.sidebar.number_input("Distance Fraction", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    if st.button("Detect Steps") and (st.session_state.x is not None) and (st.session_state.data is not None) and (st.session_state.file_path is not None):
        x, fitted_steps, optimal_step_locs, sorted_residuals = detect_steps(st.session_state.x, st.session_state.data, st.session_state.file_path, filter_window, filter_polyorder, scaling_factor, distance_fraction)

        # Show the results
        fig, ax = plt.subplots()
        ax.plot(x, st.session_state.data, label="Original Data", linewidth=0.8)
       
        ax.plot(x, fitted_steps, label="Fitted Steps", linewidth=1.5)
        ax.legend()
        ax.set_xlabel("X-axis label")
        ax.set_ylabel("Y-axis label")
        
        st.pyplot(fig)

        # Plot the quality check
        fig2, ax2 = plt.subplots()
        ax2.plot(range(len(sorted_residuals)), sorted_residuals, label="Sorted Residuals vs Iteration Steps", linewidth=1, marker='o', markersize=4)
        ax2.axvline(x=len(optimal_step_locs), color='b', linestyle='--', label="Threshold")
        ax2.legend()
        ax2.set_xlabel('Iteration Steps')
        ax2.set_ylabel('Residuals')
        st.pyplot(fig2)

        # Export the file
        export_data = pd.DataFrame({"x": x, "data": st.session_state.data, "fitted_steps": fitted_steps})
        csv_export = export_data.to_csv(index=False)
        b64 = base64.b64encode(csv_export.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="export_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
